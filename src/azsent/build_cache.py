"""Prebuild the token/feature cache for one or more backbones (runner step)."""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .lexicon import load_lexicon
from .modeling import resolve_backbone
from .tokcache import build_cache
from .utils import log, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["xlmr"])
    ap.add_argument("--backbone-local", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    setup_logging(Path(cfg.paths.runs_dir) / "logs" / "cache.log")
    from transformers import AutoTokenizer

    try:
        lex = load_lexicon(cfg.paths.data_dir, cfg.paths.lexicon_subdir)
    except FileNotFoundError as e:
        log.warning("%s - building the cache with empty features", e)
        lex = {}
    hf_cache = str(Path(cfg.paths.runs_dir) / "hf_cache")
    for key in args.backbones:
        bid = args.backbone_local or resolve_backbone(cfg.backbones[key], hf_cache)
        tok = AutoTokenizer.from_pretrained(bid, cache_dir=hf_cache, use_fast=True)
        build_cache(cfg, tok, key, lex, cfg.train.max_len, force=args.force)


if __name__ == "__main__":
    main()
