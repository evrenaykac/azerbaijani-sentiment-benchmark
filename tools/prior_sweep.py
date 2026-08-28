"""Does the domain-adaptation gain come back when the test prior is shifted?

The v1 split had a gold_test label prior of roughly 28/55/17 against a gold_train
prior of 37/19/44, and the full recipe beat plain fine-tuning by +2.5 Macro-F1
under LODO (+9.6 on the hardest domain, CI excluding zero). The v4 split matches
the prior across all three partitions to within 0.04 points, and the same gain is
-0.5 and nowhere significant.

Two explanations fit that: the v4 corpus is different, or the v1 gain was
robustness to LABEL-PRIOR shift rather than to domain shift. This separates them
without touching the corpus and without retraining anything: the already-trained
models are re-scored on test subsets resampled to a range of label priors, from
the matched prior out to the v1 prior. Everything except the test prior is held
fixed - same models, same seeds, same items, same domains.

If the gap between systems grows with prior divergence, the v1 "domain
robustness" result was a prior-shift artifact, and that is a measurable,
controlled finding rather than a null result.

    python3 tools/prior_sweep.py --runs <runs/runs> --out runs/report/prior_sweep.csv \
        --systems xlmr_ft full full_midinject xlmr_supcon mdeberta_ft mdeberta_dapt
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
                    help="v4 gold prior (neg neu pos)")
    ap.add_argument("--shifted", nargs=3, type=float, default=[0.28, 0.55, 0.17],
                    help="v1 gold_test prior (neg neu pos)")
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
        raise SystemExit(f"{a.regime} rejiminde {a.systems} icin kosu bulunamadi: {root}")

    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)

    def show(df, title):
        piv = df.groupby(["t", "l1_from_matched", "system"])["macro_f1"].mean().unstack()
        print(f"\n=== {title}: macro-F1 ===")
        print(piv.round(4).to_string())
        if a.baseline in piv.columns:
            print(f"--- {a.baseline}'e gore fark (puan) ---")
            print(((piv.drop(columns=[a.baseline]).sub(piv[a.baseline], axis=0)) * 100).round(2).to_string())
            # the number the argument turns on: how much of the gap the prior moves
            d0 = (piv.iloc[0] - piv.iloc[0][a.baseline]) * 100
            d1 = (piv.iloc[-1] - piv.iloc[-1][a.baseline]) * 100
            print(f"--- eslesmis oncul -> kaymis oncul, farkin degisimi (puan) ---")
            print((d1 - d0).drop(a.baseline).round(2).to_string())

    show(out, "TUM ALANLAR (ortalama)")
    for sc in sorted(out["scope"].unique()):
        show(out[out["scope"] == sc], sc)
    print(f"\nyazildi: {a.out} ({len(out)} satir)")


if __name__ == "__main__":
    main()
