"""Post-processing sweep: probes + calibration for every finished run that
doesn't have them yet. UMAP plots only for key pooled/full runs (cost)."""
from __future__ import annotations

import argparse
from pathlib import Path

from .calibrate import run_calibration
from .config import load_config, runs_root
from .probes import run_probes
from .utils import log, setup_logging

UMAP_FOR = ("pooled.all.full.", "pooled.all.xlmr_ft.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    setup_logging(Path(cfg.paths.runs_dir) / "logs" / "postproc.log")
    n_p = n_c = 0
    for run_dir in sorted(runs_root(cfg).iterdir()):
        if not (run_dir / "DONE.json").exists():
            continue
        try:
            if (run_dir / "embeddings.npz").exists() and (args.force or not (run_dir / "probes.json").exists()):
                with_umap = any(run_dir.name.startswith(p) for p in UMAP_FOR)
                run_probes(run_dir, with_umap=with_umap)
                n_p += 1
            if (run_dir / "preds_dev.csv").exists() and (args.force or not (run_dir / "calibration.json").exists()):
                run_calibration(run_dir)
                n_c += 1
        except Exception as e:  # noqa: BLE001
            log.warning("postproc failed for %s: %s", run_dir.name, e)
    log.info("postproc: %d probes, %d calibrations computed", n_p, n_c)


if __name__ == "__main__":
    main()
