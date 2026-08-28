"""Unified supervised trainer for every encoder system / regime / seed.

Examples
  python -m azsent.train_clf --system full --regime lodo --scope Public --seed 13
  python -m azsent.train_clf --system full --regime pooled --scope all \
      --silver-frac 0.5 --seed 42 --tag frac50
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import load_config, load_systems, prep_dir, runs_root
from .dataset import (CachedCollator, CachedDataset, Collator, DomainStratifiedBatchSampler,
                      LengthGroupedBatchSampler, SentDataset, DOMAINS, DOM2ID, LABEL2ID)
from .evaluate import full_report, macro_f1
from .features import FeatureExtractor
from .lexicon import load_lexicon
from .modeling import SentModel, SupConLoss, resolve_backbone
from .pools import make_pools
from .thermal import ThermalGuard
from .utils import Timer, env_info, is_done, log, mark_done, set_seed, setup_logging, write_json


def build_run_id(args) -> str:
    rid = f"{args.regime}.{args.scope}.{args.system}.s{args.seed}"
    if args.tag:
        rid += f".{args.tag}"
    return rid


def get_tokenizer(backbone_id: str, local_dir: str | None, cache_dir: str | None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(local_dir or backbone_id, cache_dir=cache_dir, use_fast=True)


@torch.no_grad()
def run_eval(model, loader, device, amp_dtype):
    model.eval()
    logits_all, emb_all, y_all, d_all, uid_all = [], [], [], [], []
    for batch in loader:
        uids = batch.pop("uids")
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(batch["input_ids"], batch["attention_mask"], feats=batch.get("feats"),
                        token_type_ids=batch.get("token_type_ids"))
        logits_all.append(out["logits"].float().cpu())
        emb_all.append(out["embedding"].float().cpu())
        y_all.append(batch["labels"].cpu())
        d_all.append(batch["domains"].cpu())
        uid_all.extend(uids)
    return (torch.cat(logits_all).numpy(), torch.cat(emb_all).numpy(),
            torch.cat(y_all).numpy(), torch.cat(d_all).numpy(), uid_all)


def _select(model, dv_loader, device, amp_dtype) -> float:
    logits, _, y, _, _ = run_eval(model, dv_loader, device, amp_dtype)
    model.train()
    return macro_f1(y, logits.argmax(1))


def train(args) -> dict:
    cfg = load_config()
    systems = load_systems()
    sysdef = dict(systems.systems[args.system])
    run_id = args.run_id or build_run_id(args)
    run_dir = runs_root(cfg) / run_id
    if is_done(run_dir) and not args.force:
        log.info("SKIP %s (already done)", run_id)
        return {"run_id": run_id, "skipped": True}
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")
    set_seed(args.seed)
    t_all = Timer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg.train.amp if device.type == "cuda" else "off"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    if str(sysdef.get("backbone", "")).lower() in [str(x).lower() for x in
                                                  cfg.train.get("force_fp32_backbones", [])]:
        amp_dtype = None   # DeBERTa-v3 yields non-finite losses under bf16/fp16
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        frac = float(cfg.train.get("vram_fraction", 0) or 0)
        if 0 < frac < 1:
            try:
                torch.cuda.set_per_process_memory_fraction(frac)
            except Exception:
                pass

    # ---- data ----------------------------------------------------------------
    train_df, dev_df, test_df, comp = make_pools(
        cfg, args.regime, args.scope,
        silver_frac=args.silver_frac, pool_mode=args.pool_mode,
        strict_lodo=args.strict_lodo, seed=args.seed,
    )
    if args.limit_train and len(train_df) > args.limit_train:
        train_df = train_df.sample(n=args.limit_train, random_state=args.seed).reset_index(drop=True)
        comp["train_limited_to"] = args.limit_train
    log.info("Run %s | train=%d dev=%d test=%d | %s", run_id, len(train_df), len(dev_df), len(test_df), sysdef)

    inject = sysdef.get("inject", "none")
    with_feats = inject != "none"
    fx, lex = None, {}
    if with_feats:
        lex = load_lexicon(cfg.paths.data_dir, cfg.paths.lexicon_subdir) if not args.no_lexicon else {}
        fx = FeatureExtractor(lex, lang=args.lang, subset=sysdef.get("feats", "all"),
                              match=str(cfg.get("lexicon", {}).get("match", "prefix")),
                              min_prefix_len=int(cfg.get("lexicon", {}).get("min_prefix_len", 4)))

    cache_dir = str(Path(cfg.paths.runs_dir) / "hf_cache")
    backbone_cfg = cfg.backbones[sysdef["backbone"]]
    backbone_id = args.backbone_local or resolve_backbone(backbone_cfg, cache_dir)

    dapt_local = None
    if sysdef.get("dapt") and not args.no_dapt:
        scope_key = "all" if args.regime in ("indomain", "pooled") else f"lodo_{args.scope}"
        cand = Path(cfg.paths.runs_dir) / "dapt" / f"{sysdef['backbone']}_{scope_key}" / "model"
        if not cand.exists():
            raise FileNotFoundError(f"DAPT checkpoint missing: {cand}. Run train_dapt first (runner orders this).")
        dapt_local = str(cand)

    tok = get_tokenizer(backbone_id, args.backbone_local, cache_dir)

    # ---- token/feature cache: computed once per (tokenizer, max_len), reused
    # by every run in the queue. Falls back to on-the-fly tokenization.
    tcache = None
    if not args.no_cache:
        try:
            from .tokcache import TokenCache, build_cache

            lex_for_cache = lex if with_feats else (
                load_lexicon(cfg.paths.data_dir, cfg.paths.lexicon_subdir) if not args.no_lexicon else {})
            cpath = build_cache(cfg, tok, sysdef["backbone"], lex_for_cache, cfg.train.max_len)
            tcache = TokenCache(cpath)
        except Exception as e:  # noqa: BLE001
            log.warning("Token cache unavailable (%s) - falling back to on-the-fly tokenization", e)
            tcache = None

    if tcache is not None:
        coll = CachedCollator(tok.pad_token_id, with_feats, sysdef.get("feats", "all"))
        tr_ds = CachedDataset(train_df, tcache, with_feats)
        dv_ds = CachedDataset(dev_df, tcache, with_feats)
        te_ds = CachedDataset(test_df, tcache, with_feats)
    else:
        coll = Collator(tok, max_len=cfg.train.max_len, with_feats=with_feats)
        tr_ds = SentDataset(train_df, fx)
        dv_ds = SentDataset(dev_df, fx)
        te_ds = SentDataset(test_df, fx)

    use_supcon = bool(sysdef.get("supcon"))
    sampler_mode = sysdef.get("sampler", "shuffle")
    nw = int(cfg.train.num_workers) if not args.smoke else 0
    if use_supcon and sampler_mode in ("stratified", "indomain", "random"):
        bsampler = DomainStratifiedBatchSampler(tr_ds.label, tr_ds.domain, cfg.train.batch_size,
                                                mode=sampler_mode, seed=args.seed,
                                                lengths=tr_ds.lengths)
    else:
        bsampler = LengthGroupedBatchSampler(tr_ds.lengths, cfg.train.batch_size, seed=args.seed)
    tr_loader = DataLoader(tr_ds, batch_sampler=bsampler, collate_fn=coll, num_workers=nw,
                           persistent_workers=nw > 0)
    dv_sampler = LengthGroupedBatchSampler(dv_ds.lengths, cfg.train.eval_batch_size, seed=0)
    te_sampler = LengthGroupedBatchSampler(te_ds.lengths, cfg.train.eval_batch_size, seed=0)
    dv_loader = DataLoader(dv_ds, batch_sampler=dv_sampler, collate_fn=coll, num_workers=0)
    te_loader = DataLoader(te_ds, batch_sampler=te_sampler, collate_fn=coll, num_workers=0)

    # ---- model ---------------------------------------------------------------
    model = SentModel(
        backbone_id,
        inject=inject,
        inject_point=sysdef.get("inject_point", "post"),
        mid_layer=int(sysdef.get("mid_layer", 6)),
        dann=bool(sysdef.get("dann")),
        cache_dir=cache_dir,
        local_dir=dapt_local or args.backbone_local,
    ).to(device)

    epochs = args.epochs or cfg.train.epochs
    steps_per_epoch = max(1, len(tr_loader))
    max_steps = int(cfg.train.get("max_steps", 0) or 0)
    total_steps = steps_per_epoch * epochs
    if max_steps:
        total_steps = min(total_steps, max_steps)
    eval_every = int(cfg.train.get("eval_every_steps", 0) or 0)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    warmup = int(total_steps * cfg.train.warmup_ratio)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    # Optional class weighting. The gold partitions share one label prior, but the
    # human-labelled bulk layer dominates the pool by volume and carries its own
    # (neutral-lighter) prior, leaving a measured ~6 point gap on the neutral class.
    # Off by default so the objective stays comparable across systems; turn it on
    # only as a reported variant.
    cw = None
    if str(cfg.train.get("class_weight", "none")).lower() == "balanced":
        counts = train_df["label"].map(LABEL2ID).value_counts().reindex(range(3)).fillna(0)
        inv = (counts.sum() / (3 * counts.clip(lower=1))).to_numpy(dtype="float32")
        cw = torch.tensor(inv, device=device)
        log.info("class_weight=balanced -> %s", inv.round(3).tolist())
    ce = torch.nn.CrossEntropyLoss(weight=cw)
    supcon = SupConLoss(tau=args.supcon_tau if args.supcon_tau is not None else cfg.train.supcon_tau)
    lam = args.supcon_lambda if args.supcon_lambda is not None else cfg.train.supcon_lambda

    guard = ThermalGuard(cfg.thermal, run_dir / "thermal.csv")
    best_f1, best_state, best_epoch, bad = -1.0, None, -1, 0
    step = 0
    stop_training = False
    log_every = 100
    t_last = time.time()
    log.info("steps/epoch=%d, total=%d", steps_per_epoch, total_steps)
    for epoch in range(epochs):
        if bsampler is not None:
            bsampler.set_epoch(epoch)
        model.train()
        ep_loss, nb = 0.0, 0
        for batch in tr_loader:
            batch.pop("uids")
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            p = step / max(1, total_steps)
            grl = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0 if sysdef.get("dann") else 0.0
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(batch["input_ids"], batch["attention_mask"], feats=batch.get("feats"),
                            token_type_ids=batch.get("token_type_ids"), grl_lambda=grl)
                loss = ce(out["logits"].float(), batch["labels"])
                if use_supcon:
                    loss = loss + lam * supcon(out["embedding"], batch["labels"])
                if sysdef.get("dann"):
                    loss = loss + 0.1 * torch.nn.functional.cross_entropy(out["domain_logits"].float(), batch["domains"])
            if cfg.train.get("abort_on_nonfinite_loss", True) and not torch.isfinite(loss):
                raise RuntimeError(f"{run_id}: egitim iraksadi (sonlu olmayan kayip), step {step}")
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optim.step()
            sched.step()
            ep_loss += float(loss.detach())
            nb += 1
            step += 1
            if step % log_every == 0:
                dt = time.time() - t_last
                sps = log_every / max(dt, 1e-6)
                log.info("  step %d/%d | loss %.4f | %.2f step/s | ETA %.0f dk", step, total_steps,
                         ep_loss / max(1, nb), sps, (total_steps - step) / max(sps, 1e-9) / 60)
                t_last = time.time()
            if step % max(1, int(cfg.thermal.check_every_steps)) == 0:
                guard.maybe_pause()
            if eval_every and step % eval_every == 0:
                f1 = _select(model, dv_loader, device, amp_dtype)
                log.info("  step %d | dev macro-F1 %.4f (best %.4f)", step, f1, max(f1, best_f1))
                if f1 > best_f1 + 1e-5:
                    best_f1, best_epoch, bad = f1, epoch, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= cfg.train.patience:
                        log.info("Early stop at step %d (no dev gain in %d evaluations)", step, bad)
                        stop_training = True
                        break
                t_last = time.time()
            if step >= total_steps:
                log.info("Reached the step cap (%d)", total_steps)
                stop_training = True
                break
        if stop_training:
            break
        if not eval_every:
            f1 = _select(model, dv_loader, device, amp_dtype)
            log.info("epoch %d | loss %.4f | dev macro-F1 %.4f", epoch, ep_loss / max(1, nb), f1)
            if f1 > best_f1 + 1e-5:
                best_f1, best_epoch, bad = f1, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.train.patience:
                    log.info("Early stop at epoch %d", epoch)
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    if args.save_model:
        torch.save(model.state_dict(), run_dir / "model.pt")

    # ---- final evaluation ----------------------------------------------------
    dv_logits, dv_emb, dv_y, dv_d, dv_uid = run_eval(model, dv_loader, device, amp_dtype)
    te_logits, te_emb, te_y, te_d, te_uid = run_eval(model, te_loader, device, amp_dtype)
    te_pred = te_logits.argmax(1)
    report = full_report(te_y, te_pred, [DOMAINS[i] for i in te_d])

    probs = torch.softmax(torch.from_numpy(te_logits), dim=1).numpy()
    pd.DataFrame({
        "uid": te_uid, "domain": [DOMAINS[i] for i in te_d], "gold": te_y, "pred": te_pred,
        "logit_neg": te_logits[:, 0], "logit_neu": te_logits[:, 1], "logit_pos": te_logits[:, 2],
        "p_neg": probs[:, 0], "p_neu": probs[:, 1], "p_pos": probs[:, 2],
    }).to_csv(run_dir / "preds_test.csv", index=False)
    dprobs = torch.softmax(torch.from_numpy(dv_logits), dim=1).numpy()
    pd.DataFrame({
        "uid": dv_uid, "domain": [DOMAINS[i] for i in dv_d], "gold": dv_y, "pred": dv_logits.argmax(1),
        "logit_neg": dv_logits[:, 0], "logit_neu": dv_logits[:, 1], "logit_pos": dv_logits[:, 2],
        "p_neg": dprobs[:, 0], "p_neu": dprobs[:, 1], "p_pos": dprobs[:, 2],
    }).to_csv(run_dir / "preds_dev.csv", index=False)
    np.savez_compressed(run_dir / "embeddings.npz",
                        test_emb=te_emb.astype(np.float16), test_uid=np.array(te_uid),
                        test_y=te_y, test_d=te_d,
                        dev_emb=dv_emb.astype(np.float16), dev_uid=np.array(dv_uid),
                        dev_y=dv_y, dev_d=dv_d)

    metrics = {
        "run_id": run_id, "system": args.system, "regime": args.regime, "scope": args.scope,
        "seed": args.seed, "tag": args.tag, "backbone_resolved": backbone_id,
        "dapt_init": dapt_local, "system_flags": sysdef,
        "hyperparams": {"lr": cfg.train.lr, "batch_size": cfg.train.batch_size, "max_len": cfg.train.max_len,
                        "epochs": epochs, "max_steps": max_steps, "eval_every_steps": eval_every,
                        "patience_evals": cfg.train.patience, "steps_run": step,
                        "warmup_ratio": cfg.train.warmup_ratio, "weight_decay": cfg.train.weight_decay,
                        "supcon_lambda": lam if use_supcon else None,
                        "supcon_tau": (args.supcon_tau if args.supcon_tau is not None else cfg.train.supcon_tau) if use_supcon else None,
                        "sampler": sampler_mode if use_supcon else "shuffle",
                        "silver_frac": args.silver_frac, "pool_mode": args.pool_mode,
                        "strict_lodo": args.strict_lodo},
        "composition": comp,
        "dev_best_macro_f1": best_f1, "best_epoch": best_epoch,
        "test": report,
        "env": env_info(),
        "thermal": guard.summary(),
        "wall_seconds": round(t_all.elapsed(), 1),
    }
    write_json(run_dir / "metrics.json", metrics)
    mark_done(run_dir, {"test_macro_f1": report["macro_f1"]})
    log.info("DONE %s | test macro-F1 %.4f | %.0fs", run_id, report["macro_f1"], t_all.elapsed())
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--regime", required=True, choices=["indomain", "lodo", "pooled"])
    ap.add_argument("--scope", required=True, help="domain name, LODO target, or 'all'")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--silver-frac", type=float, default=None,
                    help="varsayilan config: train.default_silver_frac")
    ap.add_argument("--pool-mode", default="gold+bulk", choices=["gold+bulk", "gold_only", "bulk_only"])
    ap.add_argument("--strict-lodo", action="store_true")
    ap.add_argument("--supcon-lambda", type=float, default=None)
    ap.add_argument("--supcon-tau", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--lang", default="az")
    ap.add_argument("--no-lexicon", action="store_true")
    ap.add_argument("--no-dapt", action="store_true")
    ap.add_argument("--backbone-local", default=None, help="local model dir override (smoke tests)")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="disable the token/feature cache")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
