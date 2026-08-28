"""Calibration: ECE (15 equal-mass bins), MCE, Brier, temperature scaling
(fit on gold-dev, applied to gold-test), class-wise and per-domain views
(Reviewer 2: neutral-class calibration / uncertainty analysis)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import log, setup_logging, write_json


def _softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece_mce(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> tuple[float, float]:
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y).astype(float)
    order = np.argsort(conf)
    bins = np.array_split(order, n_bins)  # equal-mass bins
    ece, mce = 0.0, 0.0
    n = len(y)
    for b in bins:
        if len(b) == 0:
            continue
        gap = abs(conf[b].mean() - correct[b].mean())
        ece += (len(b) / n) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[y]
    return float(((probs - onehot) ** 2).sum(1).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Grid-refined NLL minimization (deterministic)."""
    def nll(T):
        p = _softmax(logits, T)
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()

    Ts = np.arange(0.5, 5.01, 0.05)
    T0 = float(Ts[int(np.argmin([nll(t) for t in Ts]))])
    Ts2 = np.arange(max(0.1, T0 - 0.05), T0 + 0.051, 0.005)
    return float(Ts2[int(np.argmin([nll(t) for t in Ts2]))])


def classwise_ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    out = {}
    names = ["negative", "neutral", "positive"]
    for c in range(probs.shape[1]):
        p = probs[:, c]
        t = (y == c).astype(float)
        order = np.argsort(p)
        bins = np.array_split(order, n_bins)
        e = 0.0
        for b in bins:
            if len(b):
                e += (len(b) / len(y)) * abs(p[b].mean() - t[b].mean())
        out[names[c]] = float(e)
    return out


def run_calibration(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    dev = pd.read_csv(run_dir / "preds_dev.csv")
    test = pd.read_csv(run_dir / "preds_test.csv")
    lcols = ["logit_neg", "logit_neu", "logit_pos"]
    dv_logits, dv_y = dev[lcols].to_numpy(), dev["gold"].to_numpy()
    te_logits, te_y = test[lcols].to_numpy(), test["gold"].to_numpy()
    T = fit_temperature(dv_logits, dv_y)
    p_before = _softmax(te_logits, 1.0)
    p_after = _softmax(te_logits, T)
    e0, m0 = ece_mce(p_before, te_y)
    e1, m1 = ece_mce(p_after, te_y)
    out = {
        "temperature": T,
        "ece_before": e0, "ece_after": e1,
        "mce_before": m0, "mce_after": m1,
        "brier_before": brier(p_before, te_y), "brier_after": brier(p_after, te_y),
        "classwise_ece_before": classwise_ece(p_before, te_y),
        "classwise_ece_after": classwise_ece(p_after, te_y),
        "by_domain": {},
    }
    for d, part in test.groupby("domain"):
        lg, yy = part[lcols].to_numpy(), part["gold"].to_numpy()
        eb, _ = ece_mce(_softmax(lg, 1.0), yy)
        ea, _ = ece_mce(_softmax(lg, T), yy)
        out["by_domain"][d] = {"ece_before": eb, "ece_after": ea, "n": int(len(part))}
    write_json(run_dir / "calibration.json", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    setup_logging()
    r = run_calibration(args.run_dir)
    log.info("calibration: T=%.3f ECE %.4f->%.4f", r["temperature"], r["ece_before"], r["ece_after"])


if __name__ == "__main__":
    main()
