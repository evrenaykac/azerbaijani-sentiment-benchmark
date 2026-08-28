"""Metric helpers shared by trainer, stats and report modules."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

LABELS = ["negative", "neutral", "positive"]


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0))


def per_class_f1(y_true, y_pred) -> dict:
    f = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {LABELS[i]: float(f[i]) for i in range(3)}


def confusion(y_true, y_pred, normalize: str | None = "true") -> list:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2], normalize=normalize)
    return np.round(cm, 4).tolist()


def full_report(y_true, y_pred, domains=None) -> dict:
    out = {
        "macro_f1": macro_f1(y_true, y_pred),
        "per_class_f1": per_class_f1(y_true, y_pred),
        "confusion_row_norm": confusion(y_true, y_pred),
        "confusion_counts": confusion(y_true, y_pred, normalize=None),
        "n": int(len(y_true)),
    }
    if domains is not None:
        domains = np.asarray(domains)
        by_dom = {}
        for d in np.unique(domains):
            m = domains == d
            by_dom[str(d)] = {
                "macro_f1": macro_f1(np.asarray(y_true)[m], np.asarray(y_pred)[m]),
                "per_class_f1": per_class_f1(np.asarray(y_true)[m], np.asarray(y_pred)[m]),
                "confusion_row_norm": confusion(np.asarray(y_true)[m], np.asarray(y_pred)[m]),
                "n": int(m.sum()),
            }
        out["by_domain"] = by_dom
        out["macro_f1_domain_avg_unweighted"] = float(np.mean([v["macro_f1"] for v in by_dom.values()]))
    return out
