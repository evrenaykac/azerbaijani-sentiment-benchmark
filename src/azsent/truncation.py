"""Truncation report (Reviewer 1, minor item 3): per-domain share of comments
truncated at max_len subword tokens, plus token-mass lost, per tokenizer."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, prep_dir
from .modeling import resolve_backbone
from .utils import log, setup_logging, write_json


def truncation_report(backbone_keys=("xlmr", "mdeberta"), max_len: int = 128,
                      backbone_local: str | None = None) -> dict:
    from transformers import AutoTokenizer

    cfg = load_config()
    p = prep_dir(cfg)
    bulk = pd.read_parquet(p / "bulk.parquet")
    gold = pd.read_parquet(p / "gold.parquet")
    gold = gold.rename(columns={"final_label": "label"})
    frames = [bulk[["text", "domain"]], gold[["text", "domain"]]]
    df = pd.concat(frames, ignore_index=True)
    cache_dir = str(Path(cfg.paths.runs_dir) / "hf_cache")
    out = {"max_len": max_len, "by_tokenizer": {}}
    keys = ["local"] if backbone_local else backbone_keys
    for key in keys:
        bid = backbone_local or resolve_backbone(cfg.backbones[key], cache_dir)
        tok = AutoTokenizer.from_pretrained(bid, cache_dir=cache_dir, use_fast=True)
        res = {}
        for dom, g in df.groupby("domain"):
            lens = [len(ids) for ids in tok(g["text"].tolist(), truncation=False, padding=False)["input_ids"]]
            lens = np.array(lens)
            res[dom] = {
                "n": int(len(lens)),
                "mean_subwords": float(lens.mean()),
                "p95_subwords": float(np.percentile(lens, 95)),
                "p99_subwords": float(np.percentile(lens, 99)),
                "share_truncated": float((lens > max_len).mean()),
                "token_mass_lost": float(np.clip(lens - max_len, 0, None).sum() / lens.sum()),
            }
        all_lens = np.array([len(ids) for ids in tok(df["text"].tolist(), truncation=False, padding=False)["input_ids"]])
        res["ALL"] = {
            "n": int(len(all_lens)),
            "share_truncated": float((all_lens > max_len).mean()),
            "token_mass_lost": float(np.clip(all_lens - max_len, 0, None).sum() / all_lens.sum()),
        }
        out["by_tokenizer"][bid] = res
        log.info("Truncation @%d for %s: overall %.3f%% comments truncated", max_len, bid,
                 100 * res["ALL"]["share_truncated"])
    write_json(p / "truncation_report.json", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--backbone-local", default=None)
    args = ap.parse_args()
    setup_logging()
    truncation_report(max_len=args.max_len, backbone_local=args.backbone_local)


if __name__ == "__main__":
    main()
