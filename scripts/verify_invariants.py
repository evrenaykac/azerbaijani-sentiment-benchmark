#!/usr/bin/env python3
"""Re-derive the four leakage invariants from the released index alone.

This script is deliberately independent of the pipeline in ``src/``: it reads
only ``data/corpus_index.csv.gz`` and re-computes, from scratch, the four
conditions reported in Section III-E of the paper. If the released split were
contaminated, this script would say so.

Because raw comment text cannot be redistributed (platform terms), the index
ships ``text_key_sha256``: the first 16 hex characters of the SHA-256 digest of
the pipeline's own deduplication key (NFKC, Azerbaijani-aware lowercasing,
URL/mention stripping, punctuation and emoji removed, whitespace squeezed).
Equality of that hash is equality of the dedup key, so invariants 1 and 3 are
verifiable without the text itself.

Usage:
    python scripts/verify_invariants.py
Exit status is 0 if all four invariants hold, 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "corpus_index.csv.gz"

EVAL_ROLES = ("gold_dev", "gold_test")


def load() -> pd.DataFrame:
    df = pd.read_csv(INDEX, dtype=str, keep_default_na=False, na_values=[""])
    for col in ("uid", "domain", "split", "text_key_sha256"):
        if col not in df.columns:
            sys.exit(f"index is missing required column: {col}")
    return df


def main() -> int:
    df = load()
    print(f"index: {len(df):,} rows from {INDEX.relative_to(ROOT)}")

    gold = df[df["gold_role"].notna()]
    train_like = df[(df["gold_role"] == "gold_train") | (df["gold_role"].isna() & (df["split"] == "train"))]
    eval_like = df[df["gold_role"].isin(EVAL_ROLES)]
    print(f"  gold partition: {len(gold):,}   training pool: {len(train_like):,}   evaluation: {len(eval_like):,}")

    failures = 0

    # (1) no duplicate normalized text anywhere in the corpus
    dup = int(df["text_key_sha256"].duplicated().sum())
    ok = dup == 0
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] 1. duplicate normalized text ....... {dup}")

    # (2) no source group spanning two partitions (train / dev / test)
    grouped = df[df["video_id_hash"].notna()].copy()
    spanning = grouped.groupby("video_id_hash")["split"].nunique()
    n_span = int((spanning > 1).sum())
    ok = n_span == 0
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] 2. source groups spanning splits ... {n_span}")

    # (3) no evaluation text in any training partition
    overlap_text = len(set(eval_like["text_key_sha256"]) & set(train_like["text_key_sha256"]))
    ok = overlap_text == 0
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] 3. evaluation text in training ..... {overlap_text}")

    # (4) no evaluation source group in any training partition
    eval_groups = set(eval_like["video_id_hash"].dropna())
    train_groups = set(train_like["video_id_hash"].dropna())
    overlap_group = len(eval_groups & train_groups)
    ok = overlap_group == 0
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] 4. evaluation group in training .... {overlap_group}")

    # Informational, not an invariant: a source group may hold rows in both the
    # gold and the bulk *training* layers, because the gold partition is drawn
    # from the same pool. Both layers are training material, so this is by
    # design and is not leakage. It is printed so that no reader has to wonder.
    grouped = grouped.copy()
    grouped["layer"] = grouped["gold_role"].fillna("bulk_" + grouped["split"])
    layer_span = grouped.groupby("video_id_hash")["layer"].nunique()
    n_layer = int((layer_span > 1).sum())
    print(f"[info ] groups spanning gold/bulk *training* layers: {n_layer} "
          f"(by design; both layers are training material)")

    # reported coverage figures, recomputed
    cov = 100.0 * df["video_id_hash"].notna().mean()
    print(f"\nsource-group identifier coverage: {cov:.1f}%  (paper reports 71.0%)")
    print("note: the remaining comments are singletons; for those, no grouping")
    print("      stronger than the text keys above can be enforced (Section III-E).")

    if failures:
        print(f"\n{failures} invariant(s) FAILED")
        return 1
    print("\nall four invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
