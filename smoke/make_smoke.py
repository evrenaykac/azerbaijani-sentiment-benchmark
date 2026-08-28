"""Build a tiny prepared-format dataset so the whole pipeline can be exercised
in minutes before committing a GPU day to it.

The smoke set is a stratified subsample of the real export, so it exercises the
same contract: canonical domains, one label prior across partitions, groups that
never straddle a split. If the pipeline breaks on this, it would have broken on
the full run - just eight hours later.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="hazir veri seti .xlsx/.parquet")
    ap.add_argument("--out", required=True, help="hedef klasor (data/ gibi kullanilir)")
    ap.add_argument("--gold-per-cell", type=int, default=12, help="alan x etiket x bolme basina gold")
    ap.add_argument("--bulk", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src = Path(a.source)
    df = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_excel(src, sheet_name="data")
    gold = df[df["gold_role"].notna()]
    # Explicit loop rather than groupby.apply: pandas 2.x stops passing the
    # grouping columns into the callable, which silently produced a smoke set
    # with zero gold rows.
    parts = [g.sample(n=min(len(g), a.gold_per_cell), random_state=a.seed)
             for _, g in gold.groupby(["gold_role", "domain", "label"])]
    keep = pd.concat(parts, ignore_index=True)
    bulk = df[df["gold_role"].isna() & (df["split"] != "test")]
    # keep whole groups so the group-integrity invariant still holds
    gids = bulk["group_id"] if "group_id" in bulk.columns else bulk["video_id_hash"].fillna("solo_" + bulk["uid"])
    picked = bulk.assign(_g=gids)
    chosen = picked["_g"].drop_duplicates().sample(frac=1.0, random_state=a.seed)
    acc, out_ids = 0, []
    sizes = picked["_g"].value_counts()
    for g in chosen:
        if acc >= a.bulk:
            break
        out_ids.append(g)
        acc += int(sizes[g])
    small = pd.concat([keep, picked[picked["_g"].isin(out_ids)].drop(columns=["_g"])], ignore_index=True)

    out = Path(a.out)
    (out / "lexicon").mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(
        [("total_rows", len(small)), ("smoke", True), ("source_file", src.name)],
        columns=["metric", "value"])
    dst = out / "smoke_dataset.xlsx"
    with pd.ExcelWriter(dst, engine="openpyxl") as w:
        small.to_excel(w, sheet_name="data", index=False)
        audit.to_excel(w, sheet_name="audit", index=False)
    print("yazildi: %s (%d satir, gold %d)" % (dst, len(small), int(small["gold_role"].notna().sum())))
    print("configs/default.yaml -> paths.dataset_name: \"smoke_dataset.xlsx\" yapip calistirin")


if __name__ == "__main__":
    main()
