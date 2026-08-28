"""Reassemble a dataset that was delivered in parts.

The export is a little over the per-file transfer limit, so it ships split by
rows. Row order does not matter to the pipeline - every split decision is
carried in the `split` / `gold_role` columns - but the parts are concatenated in
name order so the result is byte-stable.

    python3 tools/join_dataset_parts.py --parts data/*.part*.parquet --out data/dataset.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(Path(p) for p in a.parts)
    if not files:
        raise SystemExit("parca bulunamadi")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if not df["uid"].is_unique:
        raise SystemExit("birlestirme sonrasi uid benzersiz degil - parcalar cakisiyor olabilir")
    df.to_parquet(a.out, index=False)
    print("birlestirildi: %d parca -> %s (%d satir)" % (len(files), a.out, len(df)))
    print("configs/default.yaml -> paths.dataset_name: \"%s\"" % Path(a.out).name)


if __name__ == "__main__":
    main()
