"""Step 3: Frozen bulk splits + leakage guards against gold.

* Quarantines any bulk row whose dedup key OR group id collides with a
  gold-dev/gold-test instance (they may never be trained on).
* Grouped per-domain bulk train/dev partitions (dev used for DAPT early
  stopping; never for reported metrics).
* Emits frozen split id files + a counts table (R1: "train/dev/test counts by
  domain and class").
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, prep_dir
from .utils import log, setup_logging, write_json


def build_splits(cfg) -> None:
    out = prep_dir(cfg)
    corpus = pd.read_parquet(out / "corpus.parquet")
    gold = pd.read_parquet(out / "gold.parquet")

    eval_gold = gold[gold["split"].isin(["gold_dev", "gold_test"])]
    bad_keys = set(eval_gold["key"])
    bad_groups = set(g for g in eval_gold["group_id"] if g.startswith("gvid_"))
    # bulk group ids are 'vid_<id>'; gold group ids are 'gvid_<id>' - compare raw ids
    bad_group_raw = {g[5:] for g in bad_groups}
    corpus["_ovl_text"] = corpus["key"].isin(bad_keys)
    corpus["_ovl_group"] = corpus["video_id"].astype("string").isin(bad_group_raw)
    quarantine = corpus[corpus["_ovl_text"] | corpus["_ovl_group"]].copy()
    quarantine["quarantine_reason"] = np.where(quarantine["_ovl_text"], "text_matches_gold_eval", "group_matches_gold_eval")
    corpus = corpus[~(corpus["_ovl_text"] | corpus["_ovl_group"])].copy()
    corpus = corpus.drop(columns=["_ovl_text", "_ovl_group"])
    quarantine = quarantine.drop(columns=["_ovl_text", "_ovl_group"])
    if len(quarantine):
        quarantine.to_parquet(out / "bulk_quarantined.parquet", index=False)
    log.info("Quarantined %d bulk rows overlapping gold dev/test", len(quarantine))

    # also collapse text collisions between bulk and gold-train (keep the gold copy only)
    train_gold_keys = set(gold.loc[gold["split"] == "gold_train", "key"])
    dup_with_gold_train = corpus["key"].isin(train_gold_keys)
    if dup_with_gold_train.any():
        q2 = corpus[dup_with_gold_train].copy()
        q2["quarantine_reason"] = "text_matches_gold_train"
        quarantine = pd.concat([quarantine, q2], ignore_index=True)
        quarantine.to_parquet(out / "bulk_quarantined.parquet", index=False)
        corpus = corpus[~dup_with_gold_train].copy()
        log.info("Removed %d bulk rows duplicating gold-train texts", int(dup_with_gold_train.sum()))

    # --- grouped per-domain bulk train/dev -----------------------------------
    # Whole groups go to dev; a group spanning several domains is assigned as a
    # unit (its rows become dev in every domain it touches).
    rng = np.random.RandomState(cfg.splits.seed)
    corpus["bulk_split"] = "bulk_train"
    all_dev_groups: set = set()
    for dom, gdf in corpus.groupby("domain"):
        groups = gdf["group_id"].unique()
        rng.shuffle(groups)
        target = int(round(len(gdf) * cfg.splits.bulk_dev_frac))
        acc = int(gdf["group_id"].isin(all_dev_groups).sum())  # spillover from earlier domains
        counts = gdf.groupby("group_id").size()
        for gid in groups:
            if acc >= target:
                break
            if gid in all_dev_groups:
                continue
            all_dev_groups.add(gid)
            acc += int(counts[gid])
    corpus.loc[corpus["group_id"].isin(all_dev_groups), "bulk_split"] = "bulk_dev"

    corpus.to_parquet(out / "bulk.parquet", index=False)

    # --- guards ---------------------------------------------------------------
    assert not set(corpus["key"]) & set(eval_gold["key"]), "text leakage bulk<->gold eval"
    gold_groups_raw = {g[5:] for g in gold["group_id"] if g.startswith("gvid_")}
    tr = corpus[corpus["bulk_split"] == "bulk_train"]
    dv = corpus[corpus["bulk_split"] == "bulk_dev"]
    assert not set(tr["group_id"]) & set(dv["group_id"]), "group leakage bulk train<->dev"

    # --- frozen id files + counts table --------------------------------------
    sd = Path(out / "split_ids")
    sd.mkdir(exist_ok=True)
    for s, part in gold.groupby("split"):
        part[["uid", "domain", "final_label", "group_id"]].to_csv(sd / f"{s}.csv", index=False)
    for s, part in corpus.groupby("bulk_split"):
        part[["uid", "domain", "group_id", "has_human_label"]].to_csv(sd / f"{s}.csv", index=False)

    counts = {}
    for s, part in gold.groupby("split"):
        counts[s] = {f"{d}|{l}": int(c) for (d, l), c in part.groupby(["domain", "final_label"]).size().items()}
    for s, part in corpus.groupby("bulk_split"):
        counts[s] = {
            f"{d}|human:{h}": int(c)
            for (d, h), c in part.groupby(["domain", "has_human_label"]).size().items()
        }
    write_json(out / "split_counts.json", counts)

    tbl = []
    for s, part in gold.groupby("split"):
        piv = part.pivot_table(index="domain", columns="final_label", values="uid", aggfunc="count").fillna(0).astype(int)
        piv["split"] = s
        tbl.append(piv.reset_index())
    pd.concat(tbl, ignore_index=True).to_csv(out / "gold_counts_by_domain_class.csv", index=False)
    log.info("Splits frozen. Bulk: %s", corpus["bulk_split"].value_counts().to_dict())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    cfg = load_config()
    setup_logging(Path(cfg.paths.runs_dir) / "logs" / "splits.log")
    build_splits(cfg)


if __name__ == "__main__":
    main()
