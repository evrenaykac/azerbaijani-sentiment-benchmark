"""One-time download of every backbone used by the queue.

Runs with network access; afterwards the whole pipeline can run with
HF_HUB_OFFLINE=1, which removes the per-job Hub API calls entirely. Those
calls are anonymous and get rate-limited once dozens of jobs have started,
which shows up as a job that hangs for tens of minutes before its first log
line.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import load_config
from .utils import log, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+",
                    default=["xlmr", "mdeberta", "mbert", "berturk", "allma", "labse"])
    args = ap.parse_args()
    os.environ.pop("HF_HUB_OFFLINE", None)   # this step needs the network
    cfg = load_config()
    setup_logging(Path(cfg.paths.runs_dir) / "logs" / "fetch_models.log")
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

    cache = str(Path(cfg.paths.runs_dir) / "hf_cache")
    ok, failed = [], []
    for key in args.backbones:
        cands = cfg.backbones[key]
        cands = cands if isinstance(cands, (list, tuple)) else [cands]
        done = False
        for c in cands:
            try:
                log.info("downloading %s (%s) ...", key, c)
                AutoTokenizer.from_pretrained(c, cache_dir=cache, use_fast=True)
                try:
                    AutoModelForMaskedLM.from_pretrained(c, cache_dir=cache)
                except Exception:
                    AutoModel.from_pretrained(c, cache_dir=cache)
                ok.append(f"{key}={c}")
                done = True
                break
            except Exception as e:  # noqa: BLE001
                log.warning("  %s failed: %s", c, str(e)[:200])
        if not done:
            failed.append(key)
    log.info("downloaded: %s", ", ".join(ok) or "none")
    if failed:
        log.warning("NOT available: %s - jobs needing these will be skipped/fail", ", ".join(failed))


if __name__ == "__main__":
    main()
