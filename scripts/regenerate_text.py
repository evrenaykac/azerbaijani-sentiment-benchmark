#!/usr/bin/env python3
"""Rejoin comment text to the released index.

The index in ``data/`` carries every field needed to reproduce the experiments
except the comment text itself, which platform terms do not allow us to
redistribute. This script restores it from a copy you hold yourself.

The join key is not an opaque identifier but the text itself: for each row of
your file the script recomputes the pipeline's deduplication key (NFKC,
Azerbaijani-aware lowercasing, URL and mention stripping, punctuation and emoji
removed, whitespace squeezed), hashes it with SHA-256, and matches the first 16
hex characters against ``text_key_sha256``. Anyone who holds the same public
comments -- the authors, or a third party who re-collects them -- can therefore
rebuild the corpus without trusting an identifier scheme.

Usage:
    python scripts/regenerate_text.py --source my_comments.csv --text-column text
    python scripts/regenerate_text.py --source export.parquet --out corpus.parquet

The output is written only for rows that matched; unmatched index rows are
reported so that coverage is explicit rather than silent.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "corpus_index.csv.gz"

# --- the pipeline's normalization, reproduced here so this script stands alone
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_WS_RE = re.compile(r"\s+")
_REPEAT_PUNCT_RE = re.compile(r"([!?.,])\1{2,}")
_AZ_LOWER_MAP = str.maketrans({"İ": "i", "I": "ı"})


def dedup_key(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text))
    t = _URL_RE.sub(" ", t)
    t = _MENTION_RE.sub(" ", t)
    t = _REPEAT_PUNCT_RE.sub(r"\1\1", t)
    t = t.translate(_AZ_LOWER_MAP).lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    return _WS_RE.sub(" ", t).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(dedup_key(text).encode("utf-8")).hexdigest()[:16]


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=Path,
                    help="your file holding the comment text (csv, parquet or xlsx)")
    ap.add_argument("--text-column", default="text", help="name of the text column")
    ap.add_argument("--out", default=ROOT / "data" / "corpus_with_text.parquet", type=Path)
    args = ap.parse_args()

    if not args.source.exists():
        sys.exit(f"source not found: {args.source}")

    idx = pd.read_csv(INDEX, dtype=str, keep_default_na=False, na_values=[""])
    src = read_any(args.source)
    if args.text_column not in src.columns:
        sys.exit(f"column {args.text_column!r} not in source; available: {list(src.columns)[:12]}")

    print(f"index rows : {len(idx):,}")
    print(f"source rows: {len(src):,}")

    src = src[[args.text_column]].dropna().rename(columns={args.text_column: "text"})
    src["text_key_sha256"] = [text_hash(t) for t in src["text"]]
    src = src.drop_duplicates("text_key_sha256")

    merged = idx.merge(src, on="text_key_sha256", how="left")
    matched = int(merged["text"].notna().sum())
    pct = 100.0 * matched / len(idx) if len(idx) else 0.0
    print(f"matched    : {matched:,} of {len(idx):,} index rows ({pct:.2f}%)")

    if matched == 0:
        print("\nNothing matched. The usual cause is a different text column, or text that has\n"
              "already been normalized differently (this script expects the raw comment).")
        return 1

    out = merged[merged["text"].notna()].copy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".csv":
        out.to_csv(args.out, index=False)
    else:
        out.to_parquet(args.out, index=False)
    print(f"written    : {args.out} ({len(out):,} rows)")

    if matched < len(idx):
        missing = len(idx) - matched
        print(f"\n{missing:,} index rows had no match in your file. Experiments run on a partial\n"
              "corpus will not reproduce the published numbers exactly; the run ledger in\n"
              "results/ remains the reference for those.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
