"""Bring an existing export into line with the pipeline's own deduplication key.

An export can be perfectly deduplicated by its author's normalizer and still be
rejected by `azsent.ingest`, because the pipeline strips emoji and punctuation
before hashing: "Afərin", "Afərin!" and "Afərin 🙏🏻" are one key here and three
rows there. That mismatch is not cosmetic - if one copy sits in train and
another in gold-test, it is evaluation leakage of exactly the kind Reviewer 1
objected to.

This tool does the minimum needed and nothing else:
  * collapses rows sharing the pipeline's dedup key, preferring a gold
    representative (so gold partitions never shrink), then a row with a group
    id, then the longest text;
  * refuses if two GOLD rows collide (that would change the gold composition,
    which is a repack decision, not a cleanup decision - use repack_dataset.py);
  * verifies the four leakage invariants and the composition tolerances;
  * preserves every column it does not understand, annotator columns included.

  python3 tools/normalize_export.py --in <export.xlsx|parquet> --out <out.parquet>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from azsent.normalize import dedup_key  # noqa: E402

LABELS = ["negative", "neutral", "positive"]
GOLD_ROLES = ["gold_train", "gold_dev", "gold_test"]


def _read(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_excel(p, sheet_name="data")


def _groups(df: pd.DataFrame) -> pd.Series:
    g = df["video_id_hash"].astype("string")
    return g.where(g.notna() & (g.astype(str).str.len() > 0), "solo_" + df["uid"].astype(str))


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["_k"] = df["text"].astype(str).map(dedup_key)
    n_before = len(df)

    per_key_gold = df.groupby("_k")["gold_role"].apply(lambda s: s.notna().sum())
    clashes = per_key_gold[per_key_gold > 1]
    if len(clashes):
        raise SystemExit(
            f"{len(clashes)} keys carry more than one GOLD row - that calls for "
            f"repacking, not cleaning (tools/repack_dataset.py). Example keys: "
            f"{list(clashes.index[:5])}")

    # gold first, then rows carrying a real group id, then longer text; uid breaks ties
    df["_prio"] = (df["gold_role"].notna().astype(int) * 4
                   + df["video_id_hash"].notna().astype(int) * 2
                   + df["text"].astype(str).str.len() / 1e9)
    df = df.sort_values(["_k", "_prio", "uid"], ascending=[True, False, True])
    dropped = df.duplicated("_k", keep="first")
    removed = df[dropped]
    out = df[~dropped].drop(columns=["_k", "_prio"]).reset_index(drop=True)

    rep = {
        "rows_in": n_before,
        "rows_out": len(out),
        "removed": int(dropped.sum()),
        "removed_gold": int(removed["gold_role"].notna().sum()),
        "removed_by_split": removed["split"].value_counts().to_dict(),
    }
    return out, rep


def verify(df: pd.DataFrame) -> dict:
    """Re-derive every invariant independently of whatever produced the file."""
    k = df["text"].astype(str).map(dedup_key)
    grp = _groups(df)
    ev = df["split"].isin(["dev", "test"])
    tr = df["split"] == "train"
    res = {
        "duplicate_normalized_texts": int(len(k) - k.nunique()),
        "duplicate_uid": int(len(df) - df["uid"].nunique()),
        "groups_spanning_splits": int((df.assign(_g=grp).groupby("_g")["split"].nunique() > 1).sum()),
        "eval_text_in_train": len(set(k[ev]) & set(k[tr])),
        "eval_group_in_train": len(set(grp[ev]) & set(grp[tr])),
    }
    return res


def composition(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    g = df[df["gold_role"].notna()]
    lab = pd.crosstab(g["gold_role"], g["label"], normalize="index").reindex(GOLD_ROLES) * 100
    dom = pd.crosstab(g["gold_role"], g["domain"], normalize="index").reindex(GOLD_ROLES) * 100
    return lab, dom, float((lab.max() - lab.min()).max()), float((dom.max() - dom.min()).max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--label-tol", type=float, default=1.5)
    ap.add_argument("--domain-tol", type=float, default=2.0)
    a = ap.parse_args()

    df = _read(Path(a.src))
    print(f"read: {len(df)} rows, {len(df.columns)} columns")

    out, rep = deduplicate(df)
    print(f"deduplication: {rep['removed']} rows dropped "
          f"(gold {rep['removed_gold']}), remaining {rep['rows_out']}")
    print(f"  by split: {rep['removed_by_split']}")

    inv = verify(out)
    print("verification:", inv)
    bad = {k: v for k, v in inv.items() if v}
    if bad:
        raise SystemExit(f"VERIFICATION FAILED: {bad}")

    lab, dom, dlab, ddom = composition(out)
    print("\ngold label distribution (%):"); print(lab.round(2).to_string())
    print("\ngold domain distribution (%):"); print(dom.round(2).to_string())
    print(f"\nlargest gap: label {dlab:.2f} points (tolerance {a.label_tol}), "
          f"domain {ddom:.2f} points (tolerance {a.domain_tol})")
    if dlab > a.label_tol or ddom > a.domain_tol:
        raise SystemExit("composition tolerance exceeded - repack with tools/repack_dataset.py")

    dst = Path(a.dst)
    if dst.suffix == ".parquet":
        out.to_parquet(dst, index=False)
    else:
        with pd.ExcelWriter(dst, engine="openpyxl") as w:
            out.to_excel(w, sheet_name="data", index=False)
    print(f"\nwritten: {dst} ({len(out)} rows, {len(out.columns)} columns)")


if __name__ == "__main__":
    main()
