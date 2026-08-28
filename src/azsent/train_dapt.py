"""Domain-adaptive pretraining (continued MLM) on source-domain bulk text.

Scopes: 'all' (in-domain & pooled regimes) or 'lodo_<Target>' (LODO folds -
held-out target text fully excluded, Reviewer 1 leakage protocol).

Output: <runs>/dapt/<backbone>_<scope>/model  (HF save_pretrained directory)
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import load_config
from .dataset import LengthGroupedBatchSampler
from .modeling import resolve_backbone
from .pools import dapt_texts
from .thermal import ThermalGuard
from .utils import Timer, env_info, is_done, log, mark_done, set_seed, setup_logging, write_json


class MLMDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]


class MLMCollator:
    def __init__(self, tok, max_len: int, mask_prob: float):
        self.tok = tok
        self.max_len = max_len
        self.p = mask_prob

    def __call__(self, texts):
        enc = self.tok(list(texts), truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
        ids = enc["input_ids"].clone()
        labels = ids.clone()
        special = torch.zeros_like(ids, dtype=torch.bool)
        for sid in self.tok.all_special_ids:
            special |= ids == sid
        rand = torch.rand_like(ids, dtype=torch.float)
        mask = (rand < self.p) & ~special & enc["attention_mask"].bool()
        labels[~mask] = -100
        r2 = torch.rand_like(ids, dtype=torch.float)
        ids[mask & (r2 < 0.8)] = self.tok.mask_token_id
        rnd_tok = torch.randint(low=0, high=self.tok.vocab_size, size=ids.shape)
        sel = mask & (r2 >= 0.8) & (r2 < 0.9)
        ids[sel] = rnd_tok[sel]
        enc["input_ids"] = ids
        enc["labels"] = labels
        return enc



def _split_mlm(model):
    """Return (encoder, lm_head) so the vocabulary projection can be applied to
    masked positions only. XLM-R's head is 768x250002: computing it for every
    token costs ~70% of the forward pass and dominates memory, while the loss
    only ever uses the ~15% masked positions. Returns None if the architecture
    cannot be split (falls back to the standard forward)."""
    base = None
    for attr in ("roberta", "bert", "deberta", "electra", "xlm_roberta"):
        if hasattr(model, attr):
            base = getattr(model, attr)
            break
    if base is None:
        base = getattr(model, "base_model", None)
    head = None
    for attr in ("lm_head", "cls", "predictions"):
        if hasattr(model, attr):
            head = getattr(model, attr)
            break
    return (base, head) if (base is not None and head is not None) else None


def _mlm_loss(model, split, batch):
    """Cross-entropy over masked positions only (identical to the standard
    MLM loss, which ignores label -100 positions anyway)."""
    if split is None:
        return model(**batch).loss
    base, head = split
    labels = batch["labels"]
    sel = labels != -100
    if not bool(sel.any()):
        return None
    out = base(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    logits = head(hidden[sel])                      # [n_masked, vocab]
    return torch.nn.functional.cross_entropy(logits.float(), labels[sel])


@torch.no_grad()
def mlm_eval(model, loader, device, amp_dtype, max_batches: int = 50, split=None):
    model.eval()
    tot, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            l = _mlm_loss(model, split, batch)
        if l is None:
            continue
        tot += float(l)
        n += 1
    model.train()
    return tot / max(1, n)


def run_dapt(scope: str, backbone_key: str = "xlmr", steps: int | None = None,
             backbone_local: str | None = None, seed: int = 42) -> Path:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    cfg = load_config()
    out_dir = Path(cfg.paths.runs_dir) / "dapt" / f"{backbone_key}_{scope}"
    if is_done(out_dir):
        log.info("SKIP dapt %s (done)", scope)
        return out_dir / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "dapt.log")
    set_seed(seed)
    t = Timer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    if str(backbone_key).lower() in [str(x).lower() for x in
                                     cfg.train.get("force_fp32_backbones", [])]:
        amp_dtype = None   # DeBERTa-v3 is unstable under bf16/fp16 autocast
    cache_dir = str(Path(cfg.paths.runs_dir) / "hf_cache")
    backbone_id = backbone_local or resolve_backbone(cfg.backbones[backbone_key], cache_dir)
    tok = AutoTokenizer.from_pretrained(backbone_id, cache_dir=cache_dir, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(backbone_id, cache_dir=cache_dir).float().to(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        frac = float(cfg.train.get("vram_fraction", 0) or 0)
        if 0 < frac < 1:
            try:
                torch.cuda.set_per_process_memory_fraction(frac)
            except Exception:
                pass

    tr_texts, dv_texts = dapt_texts(cfg, scope)
    rng = random.Random(seed)
    rng.shuffle(tr_texts)
    dv_texts = dv_texts[: cfg.dapt.dev_sample]
    log.info("DAPT %s: %d train texts, %d dev texts, backbone %s", scope, len(tr_texts), len(dv_texts), backbone_id)

    coll = MLMCollator(tok, cfg.dapt.max_len, cfg.dapt.mask_prob)
    nw = int(cfg.train.get("num_workers", 0))
    micro = int(cfg.dapt.get("micro_batch", 8))
    accum = max(1, int(cfg.dapt.batch_size) // micro)
    # length-grouped micro-batches: MLM logits are batch x seq x 250k, so cutting
    # padding cuts both memory and compute roughly proportionally.
    tr_lens = [max(1, len(t.split())) for t in tr_texts]
    dv_lens = [max(1, len(t.split())) for t in dv_texts]
    tr_sampler = LengthGroupedBatchSampler(tr_lens, micro, seed=seed, drop_last=True)
    tr_loader = DataLoader(MLMDataset(tr_texts), batch_sampler=tr_sampler,
                           collate_fn=coll, num_workers=nw, persistent_workers=nw > 0)
    dv_loader = DataLoader(MLMDataset(dv_texts),
                           batch_sampler=LengthGroupedBatchSampler(dv_lens, micro, seed=0),
                           collate_fn=coll, num_workers=0)
    log.info("micro-batch %d x accumulation %d = effective batch %d", micro, accum, micro * accum)

    # Materialize the dev batches once under a fixed seed. With fresh random
    # masking each evaluation the dev loss jumped by more than the improvement
    # being measured, so early stopping was decided by luck.
    if cfg.dapt.get("fixed_dev_batches", True):
        torch.manual_seed(20260825)
        dv_batches = list(dv_loader)
        set_seed(seed)
    else:
        dv_batches = dv_loader
    total_steps = steps or cfg.dapt.steps
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.dapt.lr, weight_decay=0.01)
    warmup = int(total_steps * 0.06)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: s / max(1, warmup) if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    mlm_split = _split_mlm(model)
    log.info("MLM head applied to %s", "masked positions only" if mlm_split else "all positions (fallback)")
    guard = ThermalGuard(cfg.thermal, out_dir / "thermal.csv")
    best_loss, bad, step, micro_i = float("inf"), 0, 0, 0
    log_every = int(cfg.dapt.get("log_every", 50))
    t_last, running = time.time(), 0.0
    model.train()
    optim.zero_grad(set_to_none=True)
    stop = False
    ep = 0
    while not stop:
        tr_sampler.set_epoch(ep); ep += 1
        for batch in tr_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                l = _mlm_loss(model, mlm_split, batch)
                if l is None:
                    continue
                if cfg.train.get("abort_on_nonfinite_loss", True) and not torch.isfinite(l):
                    raise RuntimeError(
                        f"DAPT iraksadi ({backbone_key}/{scope}): sonlu olmayan kayip, adim {step}")
                loss = l / accum
            loss.backward()
            running += float(loss.detach()) * accum
            micro_i += 1
            if micro_i % accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1

            if step % log_every == 0:
                dt = time.time() - t_last
                sps = log_every / max(dt, 1e-6)
                eta_min = (total_steps - step) / max(sps, 1e-9) / 60
                log.info("step %d/%d | loss %.4f | %.2f step/s | ETA %.0f dk", step, total_steps,
                         running / (log_every * accum), sps, eta_min)
                running, t_last = 0.0, time.time()
            if step % max(1, int(cfg.thermal.check_every_steps)) == 0:
                guard.maybe_pause()
            if step % int(cfg.dapt.eval_every) == 0 or step >= total_steps:
                ev = mlm_eval(model, dv_batches, device, amp_dtype, split=mlm_split)
                log.info("step %d/%d | dev MLM loss %.4f (best %.4f)", step, total_steps, ev,
                         min(ev, best_loss))
                if ev < best_loss - 1e-4:
                    best_loss, bad = ev, 0
                    model.save_pretrained(out_dir / "model")
                    tok.save_pretrained(out_dir / "model")
                else:
                    bad += 1
                    if bad >= cfg.dapt.patience_evals:
                        log.info("DAPT early stop at step %d", step)
                        stop = True
                        break
                t_last = time.time()
            if step >= total_steps:
                stop = True
                break

    if not (out_dir / "model").exists():
        model.save_pretrained(out_dir / "model")
        tok.save_pretrained(out_dir / "model")
    write_json(out_dir / "dapt_metrics.json", {
        "scope": scope, "backbone": backbone_id, "steps_run": step, "best_dev_mlm_loss": best_loss,
        "n_train_texts": len(tr_texts), "protocol": {
            "objective": "masked language modeling", "mask_prob": cfg.dapt.mask_prob,
            "max_len": cfg.dapt.max_len, "batch_size": cfg.dapt.batch_size, "lr": cfg.dapt.lr,
            "target_domain_text": "excluded" if scope != "all" else "n/a (all-domain scope)",
        }, "env": env_info(), "thermal": guard.summary(), "wall_seconds": round(t.elapsed(), 1),
    })
    mark_done(out_dir)
    return out_dir / "model"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", required=True, help="'all' or 'lodo_<Target>'")
    ap.add_argument("--backbone", default="xlmr")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--backbone-local", default=None)
    args = ap.parse_args()
    run_dapt(args.scope, args.backbone, args.steps, args.backbone_local)


if __name__ == "__main__":
    main()
