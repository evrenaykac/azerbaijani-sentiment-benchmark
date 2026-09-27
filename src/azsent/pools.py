"""Training-pool assembly for every regime.

Regimes
  indomain : scope = one domain d.   train = bulk_train(d) + gold-train(d)
             dev = gold-dev(d)       test = gold-test(d)
  lodo     : scope = held-out target t.  train = bulk_train(sources) +
             gold-train(sources)     dev = gold-dev(sources)
             test = gold-test(t)
  pooled   : scope = 'all'.          train over all domains, dev/test = full
             gold-dev / gold-test (used by sensitivity analyses; labelled as
             the pooled protocol in the paper).

Silver labels
  * human-labeled bulk rows keep their labels (human-verified layer), except in
    strict mode where the fold teacher re-labels everything (Reviewer 1 item 1,
    strictest reading - reported as a robustness check).
  * unlabeled bulk rows take the scope-matched teacher's silver labels:
      indomain/pooled -> silver_all.parquet
      lodo target t   -> silver_lodo_<t>.parquet   (teacher never saw t)
  * silver_frac subsamples the silver portion (sensitivity grid).

pool_mode: gold+bulk (default) | gold_only | bulk_only
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import prep_dir
from .utils import log

ALL_DOMAINS = ["Tech", "Finance", "Social", "Retail", "Public"]


def _load(cfg):
    p = prep_dir(cfg)
    bulk = pd.read_parquet(p / "bulk.parquet")
    gold = pd.read_parquet(p / "gold.parquet")
    return bulk, gold


def silver_path(cfg, scope: str) -> Path:
    return prep_dir(cfg) / "silver" / f"silver_{scope}.parquet"


def _gold_part(gold: pd.DataFrame, split: str, domains: list[str]) -> pd.DataFrame:
    g = gold[(gold["split"] == split) & (gold["domain"].isin(domains))].copy()
    g["label"] = g["final_label"]
    g["is_gold"] = True
    return g[["uid", "text", "label", "domain", "group_id", "is_gold"]]


def make_pools(
    cfg,
    regime: str,
    scope: str,
    silver_frac: float | None = None,
    pool_mode: str = "gold+bulk",
    strict_lodo: bool = False,
    seed: int = 42,
):
    # Default comes from the config, not from a literal: the measured effect of
    # teacher-generated silver labels was -8.6 Macro-F1 under LODO, so the
    # headline pipeline is silver-free and silver volume is an ablation axis.
    if silver_frac is None:
        silver_frac = float(cfg.train.get("default_silver_frac", 0.0))
    bulk, gold = _load(cfg)
    if regime == "indomain":
        train_doms, test_doms = [scope], [scope]
        silver_scope = "all"
    elif regime == "lodo":
        train_doms = [d for d in ALL_DOMAINS if d != scope]
        test_doms = [scope]
        silver_scope = f"lodo_{scope}"
    elif regime == "pooled":
        train_doms, test_doms = ALL_DOMAINS, ALL_DOMAINS
        silver_scope = "all"
    else:
        raise ValueError(regime)

    b = bulk[(bulk["bulk_split"] == "bulk_train") & (bulk["domain"].isin(train_doms))].copy()
    sp = silver_path(cfg, silver_scope)
    silver = pd.read_parquet(sp) if sp.exists() else None

    parts = []
    comp = {}
    if pool_mode in ("gold+bulk", "bulk_only"):
        if strict_lodo:
            if silver is None:
                raise FileNotFoundError(f"strict mode needs {sp}")
            sb = b.merge(silver[["uid", "silver_label"]], on="uid", how="inner")
            sb["label"] = sb["silver_label"]
            sb["is_gold"] = False
            parts.append(sb[["uid", "text", "label", "domain", "group_id", "is_gold"]])
            comp["bulk_teacher_relabeled"] = len(sb)
        else:
            hb = b[b["has_human_label"]].copy()
            hb["is_gold"] = False
            parts.append(hb[["uid", "text", "label", "domain", "group_id", "is_gold"]])
            comp["bulk_human"] = len(hb)
            ub = b[~b["has_human_label"]].copy()
            if silver is not None and len(ub):
                ub = ub.drop(columns=["label"]).merge(
                    silver[["uid", "silver_label"]].rename(columns={"silver_label": "label"}),
                    on="uid", how="inner",
                )
                if 0.0 <= silver_frac < 1.0:
                    if silver_frac > 0:
                        ub = ub.groupby("domain", group_keys=False).sample(frac=silver_frac, random_state=seed).reset_index(drop=True)
                    else:
                        ub = ub.iloc[0:0]
                ub["is_gold"] = False
                parts.append(ub[["uid", "text", "label", "domain", "group_id", "is_gold"]])
                comp["bulk_silver"] = len(ub)
            elif silver is None:
                log.warning("No silver file %s - training without silver rows", sp)
    if pool_mode in ("gold+bulk", "gold_only"):
        gt = _gold_part(gold, "gold_train", train_doms)
        parts.append(gt)
        comp["gold_train"] = len(gt)

    train = pd.concat(parts, ignore_index=True)
    train = train[train["label"].notna()].reset_index(drop=True)
    dev = _gold_part(gold, "gold_dev", train_doms)
    test = _gold_part(gold, "gold_test", test_doms)
    comp["train_total"] = len(train)
    comp["dev"] = len(dev)
    comp["test"] = len(test)
    comp["train_by_domain_label"] = {
        f"{d}|{l}": int(c) for (d, l), c in train.groupby(["domain", "label"]).size().items()
    }
    return train, dev, test, comp


def dapt_texts(cfg, scope: str):
    """Text pools for DAPT. scope: 'all' or 'lodo_<target>'."""
    bulk, _ = _load(cfg)
    if scope == "all":
        doms = ALL_DOMAINS
    else:
        t = scope.split("_", 1)[1]
        doms = [d for d in ALL_DOMAINS if d != t]
    tr = bulk[(bulk["bulk_split"] == "bulk_train") & (bulk["domain"].isin(doms))]["text"].tolist()
    dv = bulk[(bulk["bulk_split"] == "bulk_dev") & (bulk["domain"].isin(doms))]["text"].tolist()
    return tr, dv
