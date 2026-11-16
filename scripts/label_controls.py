#!/usr/bin/env python3
"""Label and data-strategy controls on the audited split (Section VII,
Table 15 of the article).

    python scripts/label_controls.py --runs <dir with <run_id>/preds_test.csv> \
        [--conditions human:H,relabel:T,silver:S] [--regime pooled --scope all] \
        [--regime lodo --scope Public] [--n-boot 10000] [--out results/label_controls]

Conditions are run tags. H (tag ``human``) is the released data strategy, the
human-labeled bulk comments, re-run in the same environment as the other two;
T (tag ``relabel``) is the same bulk comments with the labels of a teacher
trained on gold-train (pool mode ``gold+bulk_relabeled``); S (tag ``silver``)
is H plus the unlabeled bulk comments with teacher labels (pool mode
``gold+bulk`` with ``silver_frac`` 1.0), the data strategy of Stage 2.

For each (regime, scope) cell the script writes, into ``label_controls.json``:

  * per condition and system (xlmr_ft, full): macro-F1 per seed, seed mean and
    standard deviation, per-domain means, and the training-pool composition
    and class prior recorded in metrics.json;
  * per condition: the full-minus-xlmr_ft difference under the two bootstrap
    designs of Section IV-C (seed-pair, RNG 1234; seed-mean, RNG 1235);
  * per non-H condition and system: the drop relative to H under both designs,
    and the change of the full-minus-xlmr_ft gap relative to H under a shared
    source-group resample (RNG 4321);

and, into ``runs_label_controls.csv``, one ledger row per run (no comment
text). Source groups come from data/corpus_index.csv.gz, as in
azsent.stats_tests. The teacher's agreement with the human labels on the
training rows is recorded on the machine that trained the teacher
(``relabel_control.json``, written next to this file's output), because it
needs the prepared pools rather than the predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("xlmr_ft", "full")
LABELS = ("negative", "neutral", "positive")


def macro_f1_counts(y, p):
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
    return float(f1.mean())


def load_runs(runs: Path, regime: str, scope: str, system: str, tag: str):
    suffix = f".{tag}" if tag else ""
    out = []
    for d in sorted(runs.glob(f"{regime}.{scope}.{system}.s*{suffix}")):
        if tag == "" and d.name.count(".") != 3:
            continue
        f = d / "preds_test.csv"
        if f.exists():
            out.append((d.name, pd.read_csv(f, usecols=["uid", "domain", "gold", "pred"])))
    return out


def aligned(runs_list, order):
    P, Y = [], None
    for _, df in runs_list:
        d = df.set_index("uid").loc[order]
        P.append(d["pred"].to_numpy())
        y = d["gold"].to_numpy()
        assert Y is None or (Y == y).all(), "gold labels differ between runs"
        Y = y
    return Y, np.stack(P)


def make_units(dom, grp):
    units = []
    for d in np.unique(dom):
        md = dom == d
        gids = pd.unique(grp[md])
        units.append((gids, {g: np.flatnonzero(md & (grp == g)) for g in gids}))
    return units


def resample(rng, units):
    take = []
    for gids, idx_by_g in units:
        for g in rng.choice(gids, size=len(gids), replace=True):
            take.append(idx_by_g[g])
    return np.concatenate(take)


def both_designs(y, pA, pB, units, n_boot, seed_pair_rng=1234, seed_mean_rng=1235):
    """A - B: seed-pair design (one seed pair per replicate) and seed-mean design."""
    fA = np.array([macro_f1_counts(y, p) for p in pA])
    fB = np.array([macro_f1_counts(y, p) for p in pB])
    res = {"diff_mean_over_seeds": round(float(fA.mean() - fB.mean()), 4),
           "n_seeds_A": int(len(pA)), "n_seeds_B": int(len(pB)), "n_boot": n_boot}
    rng = np.random.RandomState(seed_pair_rng)
    pairs = [(i, j) for i in range(len(pA)) for j in range(len(pB))]
    d1 = np.empty(n_boot)
    for b in range(n_boot):
        i, j = pairs[rng.randint(len(pairs))]
        ix = resample(rng, units)
        d1[b] = macro_f1_counts(y[ix], pA[i][ix]) - macro_f1_counts(y[ix], pB[j][ix])
    rng = np.random.RandomState(seed_mean_rng)
    d2 = np.empty(n_boot)
    kA, kB = len(pA), len(pB)
    for b in range(n_boot):
        ix = resample(rng, units)
        yy = y[ix]
        fa = np.array([macro_f1_counts(yy, p[ix]) for p in pA])
        fb = np.array([macro_f1_counts(yy, p[ix]) for p in pB])
        d2[b] = fa[rng.randint(kA, size=kA)].mean() - fb[rng.randint(kB, size=kB)].mean()
    for name, d in (("seed_pair", d1), ("seed_mean", d2)):
        lo, hi = np.percentile(d, [2.5, 97.5])
        res[name] = {"ci95": [round(float(lo), 4), round(float(hi), 4)],
                     "p_onesided_A_gt_B": round(float((d <= 0).mean()), 4),
                     "p_onesided_A_lt_B": round(float((d >= 0).mean()), 4),
                     "excludes_zero": bool(lo > 0 or hi < 0)}
    return res


def gap_change(y, P, units, n_boot, cond, ref="H", rng_seed=4321):
    """(full - xlmr_ft under cond) - (full - xlmr_ft under ref), one shared
    source-group resample per replicate and seeds drawn with replacement."""
    rng = np.random.RandomState(rng_seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        ix = resample(rng, units)
        yy = y[ix]
        m = {}
        for key, pm in P.items():
            f = np.array([macro_f1_counts(yy, p[ix]) for p in pm])
            m[key] = f[rng.randint(len(f), size=len(f))].mean()
        d[b] = (m[(cond, "full")] - m[(cond, "xlmr_ft")]) - (m[(ref, "full")] - m[(ref, "xlmr_ft")])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"ci95": [round(float(lo), 4), round(float(hi), 4)], "p_le_0": round(float((d <= 0).mean()), 4),
            "excludes_zero": bool(lo > 0 or hi < 0), "n_boot": n_boot}


def per_domain_means(runs_list):
    acc = {}
    for _, df in runs_list:
        for d, g in df.groupby("domain"):
            acc.setdefault(d, []).append(macro_f1_counts(g["gold"].to_numpy(), g["pred"].to_numpy()))
    return {d: round(float(np.mean(v)), 4) for d, v in sorted(acc.items())}


def pool_info(runs: Path, run_name: str):
    f = runs / run_name / "metrics.json"
    if not f.exists():
        return None
    m = json.load(open(f))
    comp = m.get("composition", {})
    tot = {l: 0 for l in LABELS}
    for k, v in comp.get("train_by_domain_label", {}).items():
        l = k.split("|", 1)[1]
        if l in tot:
            tot[l] += int(v)
    n = sum(tot.values())
    return {"pool_mode": m.get("hyperparams", {}).get("pool_mode"),
            "silver_frac": m.get("hyperparams", {}).get("silver_frac"),
            "gold_train": comp.get("gold_train"), "bulk_human": comp.get("bulk_human"),
            "bulk_human_relabeled": comp.get("bulk_human_relabeled"), "bulk_silver": comp.get("bulk_silver"),
            "train_total": comp.get("train_total"), "label_counts": tot,
            "label_pct": {l: round(100.0 * c / n, 2) for l, c in tot.items()} if n else None}


def ledger_row(runs: Path, run_name: str, cond: str):
    m = json.load(open(runs / run_name / "metrics.json"))
    comp, hp, env = m.get("composition", {}), m.get("hyperparams", {}), m.get("env", {})
    pc = m["test"].get("per_class_f1", {})
    di = m.get("dapt_init")
    return {"run_id": run_name, "condition": cond, "tag": m.get("tag", ""), "regime": m.get("regime"),
            "scope": m.get("scope"), "system": m.get("system"), "seed": m.get("seed"),
            "pool_mode": hp.get("pool_mode"), "silver_frac": hp.get("silver_frac"),
            "gold_train": comp.get("gold_train"), "bulk_human": comp.get("bulk_human"),
            "bulk_human_relabeled": comp.get("bulk_human_relabeled"), "bulk_silver": comp.get("bulk_silver"),
            "train_total": comp.get("train_total"), "macro_f1": round(m["test"]["macro_f1"], 6),
            "neg_f1": round(pc.get("negative", float("nan")), 6), "neu_f1": round(pc.get("neutral", float("nan")), 6),
            "pos_f1": round(pc.get("positive", float("nan")), 6), "n_test": m["test"].get("n"),
            "dapt_init": Path(di).parent.name if di else "", "steps_run": hp.get("steps_run"),
            "wall_seconds": m.get("wall_seconds"), "gpu": env.get("gpu")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="directory containing <run_id>/preds_test.csv")
    ap.add_argument("--conditions", default="human:H,relabel:T,silver:S", help="tag:label pairs; the first is the reference")
    ap.add_argument("--regime", action="append", default=None)
    ap.add_argument("--scope", action="append", default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default=str(ROOT / "results" / "label_controls"))
    args = ap.parse_args()
    runs = Path(args.runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    conds = [tuple(c.split(":")) for c in args.conditions.split(",")]
    ref = conds[0][1]
    cells = list(zip(args.regime or ["pooled", "lodo"], args.scope or ["all", "Public"]))
    idx = pd.read_csv(ROOT / "data" / "corpus_index.csv.gz", usecols=["uid", "video_id_hash"], dtype=str)
    grp_of = dict(zip(idx["uid"], idx["video_id_hash"].where(idx["video_id_hash"].notna(), "solo_" + idx["uid"])))

    report = {"conditions": {label: tag for tag, label in conds}, "reference": ref, "systems": list(SYSTEMS),
              "n_boot": args.n_boot, "designs": {"seed_pair": "one seed pair per replicate, RNG 1234",
                                                 "seed_mean": "seeds with replacement for both systems, RNG 1235",
                                                 "gap_change": "shared source-group resample, seeds with replacement, RNG 4321"}}
    ledger = []
    for regime, scope in cells:
        got = {}
        for tag, label in conds:
            for s in SYSTEMS:
                got[(label, s)] = load_runs(runs, regime, scope, s, tag)
        missing = [k for k, v in got.items() if not v]
        if missing:
            report[f"{regime}.{scope}"] = {"missing_runs": [f"{c}:{s}" for c, s in missing]}
            print(f"{regime}.{scope}: missing {missing}")
            continue
        order = got[(ref, "xlmr_ft")][0][1]["uid"].tolist()
        dom = got[(ref, "xlmr_ft")][0][1].set_index("uid").loc[order]["domain"].to_numpy()
        grp = np.array([grp_of.get(u, u) for u in order])
        units = make_units(dom, grp)
        P, y = {}, None
        for key, lst in got.items():
            yk, pk = aligned(lst, order)
            assert y is None or (y == yk).all()
            y, P[key] = yk, pk
        cell = {}
        for tag, label in conds:
            cell[label] = {"tag": tag}
            for s in SYSTEMS:
                f = [macro_f1_counts(y, p) for p in P[(label, s)]]
                cell[label][s] = {"runs": [n for n, _ in got[(label, s)]],
                                  "macro_f1_mean": round(float(np.mean(f)), 6),
                                  "macro_f1_sd": round(float(np.std(f, ddof=1)), 4) if len(f) > 1 else None,
                                  "macro_f1_per_seed": [round(x, 4) for x in f],
                                  "macro_f1_by_domain_mean": per_domain_means(got[(label, s)]),
                                  "pool": pool_info(runs, got[(label, s)][0][0])}
                for n, _ in got[(label, s)]:
                    ledger.append(ledger_row(runs, n, label))
            cell[label]["full_minus_xlmr_ft"] = both_designs(y, P[(label, "full")], P[(label, "xlmr_ft")], units, args.n_boot)
        for tag, label in conds[1:]:
            cell[f"{label}_minus_{ref}"] = {s: both_designs(y, P[(label, s)], P[(ref, s)], units, args.n_boot) for s in SYSTEMS}
            cell[f"gap_change_{label}_vs_{ref}"] = gap_change(y, P, units, args.n_boot, label, ref)
        report[f"{regime}.{scope}"] = cell
        summary = {label: {s: cell[label][s]["macro_f1_mean"] for s in SYSTEMS} for _, label in conds}
        print(f"{regime}.{scope}: {summary}")
    json.dump(report, open(out_dir / "label_controls.json", "w"), indent=1)
    with open(out_dir / "runs_label_controls.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ledger[0].keys()))
        w.writeheader()
        w.writerows(sorted(ledger, key=lambda r: r["run_id"]))
    print("written", out_dir / "label_controls.json", "and", out_dir / "runs_label_controls.csv")


if __name__ == "__main__":
    main()
