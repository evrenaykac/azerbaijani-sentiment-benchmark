"""Fold-safe teachers + silver-label generation (Reviewer 1, item 1).

For scope 'all': teacher trained on the full gold-train, silver labels for all
bulk-train rows -> silver_all.parquet (used by in-domain / pooled regimes).

For scope 'lodo_<Target>': teacher trained ONLY on gold-train of the four
source domains, model-selected ONLY on their gold-dev; silver labels produced
ONLY for source-domain bulk rows -> silver_lodo_<Target>.parquet. The held-out
target contributes no label information of any kind to the fold.

Each silver file covers *every* bulk-train row of its scope so that the strict
robustness mode (teacher re-labels the human layer too) needs no extra pass.
Teacher quality is measured against (a) scope gold-dev and (b) the overlapping
human-verified layer -> teacher_quality.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import load_config, prep_dir
from .dataset import Collator, SentDataset, LABEL2ID, ID2LABEL, DOMAINS
from .evaluate import full_report, macro_f1, per_class_f1
from .pools import ALL_DOMAINS, silver_path
from .thermal import ThermalGuard
from .utils import Timer, is_done, log, mark_done, set_seed, setup_logging, write_json


def _scope_domains(scope: str) -> list[str]:
    if scope == "all":
        return ALL_DOMAINS
    return [d for d in ALL_DOMAINS if d != scope.split("_", 1)[1]]


def train_teacher_and_label(scope: str, backbone_local: str | None = None, seed: int = 42,
                            epochs: int | None = None) -> Path:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = load_config()
    out_dir = Path(cfg.paths.runs_dir) / "teachers" / scope
    sp = silver_path(cfg, scope)
    if is_done(out_dir) and sp.exists():
        log.info("SKIP teacher %s (done)", scope)
        return sp
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "teacher.log")
    set_seed(seed)
    t = Timer()
    doms = _scope_domains(scope)

    p = prep_dir(cfg)
    gold = pd.read_parquet(p / "gold.parquet")
    bulk = pd.read_parquet(p / "bulk.parquet")
    gt = gold[(gold["split"] == "gold_train") & (gold["domain"].isin(doms))].copy()
    gd = gold[(gold["split"] == "gold_dev") & (gold["domain"].isin(doms))].copy()
    gt["label"], gd["label"] = gt["final_label"], gd["final_label"]
    target_rows = bulk[bulk["domain"].isin(doms)].copy()  # bulk_train + bulk_dev text rows of scope
    log.info("Teacher %s: gold-train=%d gold-dev=%d, to label=%d", scope, len(gt), len(gd), len(target_rows))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    cache_dir = str(Path(cfg.paths.runs_dir) / "hf_cache")
    backbone = backbone_local or cfg.teacher.backbone
    tok = AutoTokenizer.from_pretrained(backbone, cache_dir=cache_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(backbone, num_labels=3, cache_dir=cache_dir).to(device)

    coll = Collator(tok, max_len=cfg.train.max_len, with_feats=False)
    tr_loader = DataLoader(SentDataset(gt), batch_size=cfg.teacher.batch_size, shuffle=True, collate_fn=coll)
    dv_loader = DataLoader(SentDataset(gd), batch_size=cfg.train.eval_batch_size, collate_fn=coll)

    n_epochs = epochs or cfg.teacher.epochs
    total = max(1, len(tr_loader)) * n_epochs
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.teacher.lr, weight_decay=0.01)
    warmup = int(total * 0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: s / max(1, warmup) if s < warmup else max(0.0, (total - s) / max(1, total - warmup)))
    guard = ThermalGuard(cfg.thermal, out_dir / "thermal.csv")

    def fwd(batch):
        kw = {"input_ids": batch["input_ids"].to(device), "attention_mask": batch["attention_mask"].to(device)}
        if "token_type_ids" in batch:
            kw["token_type_ids"] = batch["token_type_ids"].to(device)
        return model(**kw).logits

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        ys, ps = [], []
        for b in loader:
            b.pop("uids")
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                lg = fwd(b)
            ys.append(b["labels"].numpy())
            ps.append(lg.float().argmax(1).cpu().numpy())
        model.train()
        return np.concatenate(ys), np.concatenate(ps)

    best_f1, best_state, bad, step = -1, None, 0, 0
    model.train()
    for ep in range(n_epochs):
        for b in tr_loader:
            b.pop("uids")
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = fwd(b)
                loss = torch.nn.functional.cross_entropy(logits.float(), b["labels"].to(device))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            step += 1
            if step % int(cfg.thermal.check_every_steps) == 0:
                guard.maybe_pause()
        y, pr = evaluate(dv_loader)
        f1 = macro_f1(y, pr)
        log.info("teacher %s epoch %d dev macro-F1 %.4f", scope, ep, f1)
        if f1 > best_f1 + 1e-5:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.teacher.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "teacher.pt")

    # ---- silver labeling ----------------------------------------------------
    model.eval()
    lab_loader = DataLoader(SentDataset(target_rows.assign(label=None)), batch_size=cfg.train.eval_batch_size,
                            collate_fn=coll, num_workers=0)
    preds, confs, uids = [], [], []
    with torch.no_grad():
        for i, b in enumerate(lab_loader):
            u = b.pop("uids")
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                lg = fwd(b)
            pb = torch.softmax(lg.float(), 1)
            preds.append(pb.argmax(1).cpu().numpy())
            confs.append(pb.max(1).values.cpu().numpy())
            uids.extend(u)
            if (i + 1) % 100 == 0:
                guard.maybe_pause()
    preds = np.concatenate(preds)
    confs = np.concatenate(confs)
    silver = pd.DataFrame({"uid": uids, "silver_label": [ID2LABEL[int(x)] for x in preds],
                           "confidence": confs.astype(np.float32), "teacher_scope": scope})
    sp.parent.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(sp, index=False)

    # ---- quality report -----------------------------------------------------
    y, pr = evaluate(dv_loader)
    human = target_rows[target_rows["has_human_label"]].merge(silver, on="uid")
    hy = human["label"].map(LABEL2ID).to_numpy()
    hp = human["silver_label"].map(LABEL2ID).to_numpy()
    quality = {
        "scope": scope, "source_domains": doms, "backbone": backbone,
        "gold_dev_macro_f1": macro_f1(y, pr), "gold_dev_per_class": per_class_f1(y, pr),
        "vs_human_layer_n": int(len(human)),
        "vs_human_layer_agreement": float((hy == hp).mean()) if len(human) else None,
        "vs_human_layer_macro_f1": macro_f1(hy, hp) if len(human) else None,
        "vs_human_layer_per_class": per_class_f1(hy, hp) if len(human) else None,
        "silver_label_dist": silver["silver_label"].value_counts().to_dict(),
        "mean_confidence": float(confs.mean()),
        "n_labeled": int(len(silver)),
        "wall_seconds": round(t.elapsed(), 1),
        "thermal": guard.summary(),
    }
    write_json(out_dir / "teacher_quality.json", quality)
    mark_done(out_dir, {"gold_dev_macro_f1": best_f1})
    log.info("Teacher %s done: dev F1 %.4f, labeled %d rows", scope, best_f1, len(silver))
    return sp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", required=True, help="'all' or 'lodo_<Target>'")
    ap.add_argument("--backbone-local", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    train_teacher_and_label(args.scope, args.backbone_local, epochs=args.epochs)


if __name__ == "__main__":
    main()
