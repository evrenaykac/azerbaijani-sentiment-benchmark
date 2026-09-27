"""Statistical testing (Reviewer 1, item 7).

* Paired bootstrap over gold-test predictions (pooled and per-domain).
* Cluster-aware bootstrap: resampling units are source groups (video ids;
  singleton groups otherwise), stratified by domain - accounts for source
  clustering.
* Seed-aware testing: for multi-seed systems the bootstrap is hierarchical
  (draw a seed pair uniformly, then a cluster resample), so the reported CI
  reflects both seed and sampling variability.
* McNemar's test on pooled predictions.
* Holm-Bonferroni adjustment across each table's family of comparisons.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, prep_dir, runs_root
from .evaluate import macro_f1
from .utils import log, read_json, setup_logging, write_json


def _load_preds(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "preds_test.csv")
    return df[["uid", "domain", "gold", "pred"]]


def _group_map(cfg) -> dict:
    gold = pd.read_parquet(prep_dir(cfg) / "gold.parquet")
    return dict(zip(gold["uid"], gold["group_id"]))


def paired_bootstrap(dfA: pd.DataFrame, dfB: pd.DataFrame, group_map: dict, n_boot: int = 10000,
                     seed: int = 1234, cluster: bool = True) -> dict:
    """P(A better than B) via bootstrap of macro-F1 difference on shared uids."""
    m = dfA.merge(dfB, on="uid", suffixes=("_a", "_b"))
    assert (m["gold_a"] == m["gold_b"]).all()
    y = m["gold_a"].to_numpy()
    pa, pb = m["pred_a"].to_numpy(), m["pred_b"].to_numpy()
    dom = m["domain_a"].to_numpy()
    grp = np.array([group_map.get(u, u) for u in m["uid"]])
    rng = np.random.RandomState(seed)
    obs = macro_f1(y, pa) - macro_f1(y, pb)

    if cluster:
        # per-domain cluster lists
        units = []
        for d in np.unique(dom):
            md = dom == d
            gids = pd.unique(grp[md])
            idx_by_g = {g: np.flatnonzero(md & (grp == g)) for g in gids}
            units.append((gids, idx_by_g))
        diffs = np.empty(n_boot)
        for b in range(n_boot):
            take = []
            for gids, idx_by_g in units:
                pick = rng.choice(gids, size=len(gids), replace=True)
                take.extend(idx_by_g[g] for g in pick)
            idx = np.concatenate(take)
            diffs[b] = macro_f1(y[idx], pa[idx]) - macro_f1(y[idx], pb[idx])
    else:
        n = len(y)
        diffs = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.randint(0, n, size=n)
            diffs[b] = macro_f1(y[idx], pa[idx]) - macro_f1(y[idx], pb[idx])

    p_onesided = float((diffs <= 0).mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(obs), "ci95": [float(lo), float(hi)],
            "p_onesided_A_gt_B": p_onesided, "n_boot": n_boot,
            "cluster_bootstrap": cluster, "n_items": int(len(y))}


def hierarchical_seed_bootstrap(runsA: list[Path], runsB: list[Path], group_map: dict,
                                n_boot: int = 10000, seed: int = 1234) -> dict:
    """Uniformly draw (seedA, seedB) then a cluster resample."""
    As = [_load_preds(r) for r in runsA]
    Bs = [_load_preds(r) for r in runsB]
    merged = []
    for a in As:
        for b in Bs:
            m = a.merge(b, on="uid", suffixes=("_a", "_b"))
            merged.append(m)
    rng = np.random.RandomState(seed)
    base = merged[0]
    dom = base["domain_a"].to_numpy()
    grp = np.array([group_map.get(u, u) for u in base["uid"]])
    units = []
    for d in np.unique(dom):
        md = dom == d
        gids = pd.unique(grp[md])
        idx_by_g = {g: np.flatnonzero(md & (grp == g)) for g in gids}
        units.append((gids, idx_by_g))
    # observed: mean over all seed pairs
    obs = float(np.mean([macro_f1(m["gold_a"], m["pred_a"]) - macro_f1(m["gold_a"], m["pred_b"]) for m in merged]))
    diffs = np.empty(n_boot)
    uid_order = base["uid"].tolist()
    aligned = []
    for m in merged:
        m = m.set_index("uid").loc[uid_order]
        aligned.append((m["gold_a"].to_numpy(), m["pred_a"].to_numpy(), m["pred_b"].to_numpy()))
    for b in range(n_boot):
        y, pa, pb = aligned[rng.randint(len(aligned))]
        take = []
        for gids, idx_by_g in units:
            pick = rng.choice(gids, size=len(gids), replace=True)
            take.extend(idx_by_g[g] for g in pick)
        idx = np.concatenate(take)
        diffs[b] = macro_f1(y[idx], pa[idx]) - macro_f1(y[idx], pb[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff_mean_over_seedpairs": obs, "ci95": [float(lo), float(hi)],
            "p_onesided_A_gt_B": float((diffs <= 0).mean()), "n_boot": n_boot,
            "n_seed_pairs": len(aligned), "hierarchical": True}


def mcnemar(dfA: pd.DataFrame, dfB: pd.DataFrame) -> dict:
    from scipy.stats import binomtest

    m = dfA.merge(dfB, on="uid", suffixes=("_a", "_b"))
    y = m["gold_a"].to_numpy()
    a_ok = (m["pred_a"].to_numpy() == y)
    b_ok = (m["pred_b"].to_numpy() == y)
    n01 = int((~a_ok & b_ok).sum())
    n10 = int((a_ok & ~b_ok).sum())
    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue * 1.0
    return {"n01": n01, "n10": n10, "p": float(min(1.0, p))}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[k] = float(min(1.0, running))
    return adj


def compare_systems(regime: str, baseline: str, systems: list[str], seeds: list[int],
                    scopes: list[str], tag: str = "", out_name: str | None = None) -> dict:
    cfg = load_config()
    gm = _group_map(cfg)
    rroot = runs_root(cfg)
    results, family_p = {}, {}
    for sys_name in systems:
        per_scope = {}
        pooledA, pooledB = [], []
        for scope in scopes:
            runsA = [rroot / f"{regime}.{scope}.{sys_name}.s{s}{('.' + tag) if tag else ''}" for s in seeds]
            runsB = [rroot / f"{regime}.{scope}.{baseline}.s{s}{('.' + tag) if tag else ''}" for s in seeds]
            runsA = [r for r in runsA if (r / "preds_test.csv").exists()]
            runsB = [r for r in runsB if (r / "preds_test.csv").exists()]
            if not runsA or not runsB:
                per_scope[scope] = {"missing": True}
                continue
            r = hierarchical_seed_bootstrap(runsA, runsB, gm, cfg.stats.bootstrap_n, cfg.stats.bootstrap_seed)
            r["mcnemar_first_seed"] = mcnemar(_load_preds(runsA[0]), _load_preds(runsB[0]))
            per_scope[scope] = r
            pooledA.extend(runsA)
            pooledB.extend(runsB)
        results[sys_name] = per_scope
        ps = [v["p_onesided_A_gt_B"] for v in per_scope.values() if "p_onesided_A_gt_B" in v]
        if ps:
            family_p[sys_name] = float(np.median(ps))
    adj = holm(family_p) if family_p else {}
    out = {"regime": regime, "baseline": baseline, "seeds": seeds, "scopes": scopes,
           "comparisons": results, "holm_adjusted_median_p": adj,
           "note": "hierarchical bootstrap: uniform seed-pair draw + domain-stratified cluster (source-group) resample"}
    name = out_name or f"stats_{regime}_{baseline}"
    write_json(Path(cfg.paths.runs_dir) / "stats" / f"{name}.json", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True)
    ap.add_argument("--baseline", default="xlmr_ft")
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--scopes", nargs="+", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    setup_logging()
    out = compare_systems(args.regime, args.baseline, args.systems, args.seeds, args.scopes, args.tag)
    log.info("stats written; Holm-adjusted: %s", out["holm_adjusted_median_p"])


if __name__ == "__main__":
    main()
