"""Reweight stored test predictions to a target label prior and recompute macro-F1.

Each test item gets a per-class weight (target prior / observed prior), so the
weighted macro-F1 is exact: no resampling, no Monte-Carlo noise, no retraining.
Everything except the test prior is held fixed - same models, same seeds, same
items, same domains.

Section VII of the article uses this machinery to test, from both directions,
whether a label-prior mismatch explains the apparent gain of the earlier
(unaudited) evaluation. Test 1 sweeps the released test partition from its
matched prior (--matched) towards the earlier split's gold-test prior
(--shifted, negative/neutral/positive = 28/55/17) and reports how far the gap
between systems moves; Test 2 applies the same weighting to the earlier split's
own test sets (results/earlier_split/prior_reweight_test2_earlier_split.json).
Neither explains the earlier gain, which Section VII traces to the
model-generated bulk labels of the earlier protocol.

    python tools/prior_sweep.py --runs <dir with run folders> --out results/tables/prior_sweep_lodo.csv \
        --regime lodo --systems full xlmr_ft --baseline xlmr_ft
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = [0, 1, 2]                       # neg, neu, pos
NAMES = ["negative", "neutral", "positive"]


def macro_f1(gold: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    """Weighted macro-F1: w is a per-item weight, so a resampled prior needs no
    actual resampling - the weights ARE the resample, with zero Monte-Carlo noise."""
    f1s = []
    for c in LABELS:
        tp = float(w[(gold == c) & (pred == c)].sum())
        fp = float(w[(gold != c) & (pred == c)].sum())
        fn = float(w[(gold == c) & (pred != c)].sum())
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def weights_for(gold: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-item weights that turn the empirical prior into `target`."""
    obs = np.array([(gold == c).mean() for c in LABELS])
    ratio = np.divide(target, obs, out=np.zeros_like(target), where=obs > 0)
    w = ratio[gold]
    return w * (len(gold) / w.sum())      # keep the effective sample size


def load_run(d: Path) -> pd.DataFrame | None:
    f = d / "preds_test.csv"
    if not f.exists():
        return None
    return pd.read_csv(f, usecols=["uid", "domain", "gold", "pred"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--regime", default="lodo", choices=["lodo", "indomain", "pooled"])
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--matched", nargs=3, type=float, default=[0.348, 0.294, 0.358],
                    help="matched prior of the released gold partition (neg neu pos)")
    ap.add_argument("--shifted", nargs=3, type=float, default=[0.28, 0.55, 0.17],
                    help="end point of the sweep; default: the earlier split's gold-test prior (neg neu pos)")
    ap.add_argument("--steps", type=int, default=11)
    ap.add_argument("--baseline", default="xlmr_ft")
    a = ap.parse_args()

    root = Path(a.runs)
    matched = np.array(a.matched, dtype=float); matched /= matched.sum()
    shifted = np.array(a.shifted, dtype=float); shifted /= shifted.sum()

    rows = []
    for d in sorted(root.iterdir()):
        parts = d.name.split(".")
        if len(parts) < 4 or parts[0] != a.regime or parts[2] not in a.systems:
            continue
        if len(parts) > 4:                        # tagged variants (strict, frac050, ...)
            continue
        df = load_run(d)
        if df is None or df.empty:
            continue
        gold = df["gold"].to_numpy(); pred = df["pred"].to_numpy()
        for i in range(a.steps):
            t = i / (a.steps - 1)
            target = (1 - t) * matched + t * shifted
            w = weights_for(gold, target)
            rows.append({
                "regime": parts[0], "scope": parts[1], "system": parts[2], "seed": parts[3],
                "t": round(t, 3),
                "l1_from_matched": round(float(np.abs(target - matched).sum()), 4),
                "prior_neg": round(float(target[0]), 4),
                "prior_neu": round(float(target[1]), 4),
                "prior_pos": round(float(target[2]), 4),
                "macro_f1": macro_f1(gold, pred, w),
            })
    if not rows:
        raise SystemExit(f"no runs of {a.systems} in the {a.regime} regime: {root}")

    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)

    def show(df, title):
        piv = df.groupby(["t", "l1_from_matched", "system"])["macro_f1"].mean().unstack()
        print(f"\n=== {title}: macro-F1 ===")
        print(piv.round(4).to_string())
        if a.baseline in piv.columns:
            print(f"--- difference from {a.baseline} (points) ---")
            print(((piv.drop(columns=[a.baseline]).sub(piv[a.baseline], axis=0)) * 100).round(2).to_string())
            # the number the argument turns on: how much of the gap the prior moves
            d0 = (piv.iloc[0] - piv.iloc[0][a.baseline]) * 100
            d1 = (piv.iloc[-1] - piv.iloc[-1][a.baseline]) * 100
            print(f"--- change of that difference, matched -> shifted prior (points) ---")
            print((d1 - d0).drop(a.baseline).round(2).to_string())

    show(out, "ALL DOMAINS (mean)")
    for sc in sorted(out["scope"].unique()):
        show(out[out["scope"] == sc], sc)
    print(f"\nwritten: {a.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
