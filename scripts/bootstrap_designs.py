#!/usr/bin/env python3
"""Recompute every released comparison under two bootstrap designs.

    python scripts/bootstrap_designs.py --runs <dir with runs/<run_id>/preds_test.csv> \
        [--n-boot 10000] [--out results/stats/seed_mean]

Inputs are the released per-item predictions (predictions.tar.gz of the GitHub
release, extracted so that <runs>/<run_id>/preds_test.csv exists) and
data/corpus_index.csv.gz, which supplies the source groups exactly as
azsent.stats_tests does. The comparison families, seeds and scopes are read
from results/stats/stats_*.json.

Designs
  seed-pair  the pre-specified design of results/stats/: each replicate draws
             one seed pair uniformly, then resamples source groups within the
             domain. Reproduced here with the original RNG seed (1234); the
             intervals match the released files to 1e-4.
  seed-mean  each replicate draws seeds with replacement for both systems,
             resamples source groups once, and takes the difference of the
             mean macro-F1 over the drawn seeds (RNG seed 1235). These are the
             files in results/stats/seed_mean/.

macro-F1 is computed from confusion counts; it equals
sklearn.metrics.f1_score(average="macro", labels=[0, 1, 2], zero_division=0).
Runtime: about 15 minutes on one CPU core for the 190 comparisons.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def macro_f1_counts(y: np.ndarray, p: np.ndarray) -> float:
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
    return float(f1.mean())


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[k] = float(min(1.0, running))
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="directory containing <run_id>/preds_test.csv")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default=str(ROOT / "results" / "stats" / "seed_mean"))
    ap.add_argument("--designs", nargs="+", default=["seed-mean"], choices=["seed-pair", "seed-mean"])
    args = ap.parse_args()
    runs = Path(args.runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = pd.read_csv(ROOT / "data" / "corpus_index.csv.gz", usecols=["uid", "video_id_hash"], dtype=str)
    grp_of = dict(zip(idx["uid"], idx["video_id_hash"].where(idx["video_id_hash"].notna(), "solo_" + idx["uid"])))

    def load(run_id: str) -> pd.DataFrame:
        return pd.read_csv(runs / run_id / "preds_test.csv", usecols=["uid", "domain", "gold", "pred"])

    for sf in sorted((ROOT / "results" / "stats").glob("stats_*.json")):
        rel = json.load(open(sf))
        regime, baseline, seeds, scopes = rel["regime"], rel["baseline"], rel["seeds"], rel["scopes"]
        results, pvals = {}, {}
        for sysname, per_scope in rel["comparisons"].items():
            results[sysname] = {}
            for scope in scopes:
                ra = [f"{regime}.{scope}.{sysname}.s{s}" for s in seeds if (runs / f"{regime}.{scope}.{sysname}.s{s}" / "preds_test.csv").exists()]
                rb = [f"{regime}.{scope}.{baseline}.s{s}" for s in seeds if (runs / f"{regime}.{scope}.{baseline}.s{s}" / "preds_test.csv").exists()]
                if not ra or not rb:
                    results[sysname][scope] = {"missing": True}
                    continue
                As, Bs = [load(r) for r in ra], [load(r) for r in rb]
                order = As[0]["uid"].tolist()

                def align(df):
                    d = df.set_index("uid").loc[order]
                    return d["gold"].to_numpy(), d["pred"].to_numpy()

                ya, yb = [align(a) for a in As], [align(b) for b in Bs]
                y = ya[0][0]
                for g, _ in ya + yb:
                    assert (g == y).all(), "gold labels differ between runs"
                pA, pB = np.stack([p for _, p in ya]), np.stack([p for _, p in yb])
                dom = As[0]["domain"].to_numpy()
                grp = np.array([grp_of.get(u, u) for u in order])
                units = []
                for d in np.unique(dom):
                    md = dom == d
                    gids = pd.unique(grp[md])
                    units.append((gids, {g: np.flatnonzero(md & (grp == g)) for g in gids}))

                fA = np.array([macro_f1_counts(y, p) for p in pA])
                fB = np.array([macro_f1_counts(y, p) for p in pB])
                entry = {"diff_mean_over_seeds": round(float(fA.mean() - fB.mean()), 4),
                         "n_seeds_A": len(pA), "n_seeds_B": len(pB), "n_boot": args.n_boot}

                def resample(rng):
                    take = []
                    for gids, idx_by_g in units:
                        for g in rng.choice(gids, size=len(gids), replace=True):
                            take.append(idx_by_g[g])
                    return np.concatenate(take)

                if "seed-pair" in args.designs:
                    rng = np.random.RandomState(1234)
                    pairs = [(i, j) for i in range(len(pA)) for j in range(len(pB))]
                    diffs = np.empty(args.n_boot)
                    for b in range(args.n_boot):
                        i, j = pairs[rng.randint(len(pairs))]
                        ix = resample(rng)
                        diffs[b] = macro_f1_counts(y[ix], pA[i][ix]) - macro_f1_counts(y[ix], pB[j][ix])
                    lo, hi = np.percentile(diffs, [2.5, 97.5])
                    entry["seed_pair"] = {"ci95": [round(float(lo), 4), round(float(hi), 4)],
                                          "p_onesided_A_gt_B": round(float((diffs <= 0).mean()), 4)}
                if "seed-mean" in args.designs:
                    rng = np.random.RandomState(1235)
                    kA, kB = len(pA), len(pB)
                    diffs = np.empty(args.n_boot)
                    for b in range(args.n_boot):
                        ix = resample(rng)
                        yy = y[ix]
                        fa = np.array([macro_f1_counts(yy, p[ix]) for p in pA])
                        fb = np.array([macro_f1_counts(yy, p[ix]) for p in pB])
                        diffs[b] = fa[rng.randint(kA, size=kA)].mean() - fb[rng.randint(kB, size=kB)].mean()
                    lo, hi = np.percentile(diffs, [2.5, 97.5])
                    entry["ci95"] = [round(float(lo), 4), round(float(hi), 4)]
                    entry["p_onesided_A_gt_B"] = round(float((diffs <= 0).mean()), 4)
                    pvals[(sysname, scope)] = entry["p_onesided_A_gt_B"]
                results[sysname][scope] = entry
                print(f"{sf.name} {sysname} {scope} {entry}", flush=True)
        if pvals:
            for (s, d), a in holm(pvals).items():
                results[s][d]["holm_adjusted_p_within_family"] = round(a, 4)
        out = {"regime": regime, "baseline": baseline, "design": " + ".join(args.designs), "comparisons": results}
        json.dump(out, open(out_dir / sf.name.replace(".json", "_seed_mean.json"), "w"), indent=1)
    print("written", out_dir)


if __name__ == "__main__":
    main()
