"""Aggregate every finished run into paper-ready tables (CSV + LaTeX rows).

Reads runs/<run_id>/metrics.json (+probes.json, calibration.json) and writes
<runs>/report/: one CSV per table plus a master runs.csv. Both unweighted
domain averages and pooled example-level metrics are reported (Reviewer 1,
minor item 4).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, load_systems, prep_dir, runs_root
from .utils import log, read_json, setup_logging

DOMAINS = ["Tech", "Finance", "Social", "Retail", "Public"]


def collect(cfg) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(runs_root(cfg).iterdir()):
        mj = run_dir / "metrics.json"
        if not mj.exists():
            continue
        try:
            m = read_json(mj)
        except Exception:
            continue
        if "test" not in m:
            continue
        row = {
            "run_id": m.get("run_id", run_dir.name),
            "system": m.get("system"), "regime": m.get("regime"), "scope": m.get("scope"),
            "seed": m.get("seed"), "tag": m.get("tag", ""),
            "macro_f1": m["test"]["macro_f1"],
            "neg_f1": m["test"]["per_class_f1"]["negative"],
            "neu_f1": m["test"]["per_class_f1"]["neutral"],
            "pos_f1": m["test"]["per_class_f1"]["positive"],
            "n_test": m["test"]["n"],
            "backbone": m.get("backbone_resolved"),
            "wall_seconds": m.get("wall_seconds"),
            "train_total": (m.get("composition") or {}).get("train_total"),
            "silver_frac": (m.get("hyperparams") or {}).get("silver_frac"),
            "pool_mode": (m.get("hyperparams") or {}).get("pool_mode"),
            "strict_lodo": (m.get("hyperparams") or {}).get("strict_lodo"),
            "supcon_lambda": (m.get("hyperparams") or {}).get("supcon_lambda"),
            "supcon_tau": (m.get("hyperparams") or {}).get("supcon_tau"),
            "max_temp_c": (m.get("thermal") or {}).get("max_temp_c"),
        }
        pj = run_dir / "probes.json"
        if pj.exists():
            p = read_json(pj)
            for k in ("nn@1", "nn@5", "nn@10", "nn@20", "nmi", "ari", "silhouette", "davies_bouldin",
                      "anisotropy", "twonn_id", "centroid_within_class_cross_domain", "centroid_cross_class"):
                row[k.replace("@", "_at_")] = p.get(k)
        cj = run_dir / "calibration.json"
        if cj.exists():
            c = read_json(cj)
            for k in ("temperature", "ece_before", "ece_after", "mce_before", "mce_after",
                      "brier_before", "brier_after"):
                row[k] = c.get(k)
            row["neu_ece_before"] = (c.get("classwise_ece_before") or {}).get("neutral")
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(mean: float, std: float | None) -> str:
    if std is None or np.isnan(std):
        return f"{100 * mean:.1f}"
    return f"{100 * mean:.1f}$\\pm${100 * std:.1f}"


def table_regime(df: pd.DataFrame, regime: str, out_dir: Path, systems_order: list[str]) -> None:
    d = df[(df["regime"] == regime) & (df["tag"] == "")]
    if d.empty:
        return
    recs = []
    for system, g in d.groupby("system"):
        rec = {"system": system}
        per_scope_means = []
        for scope in DOMAINS:
            gs = g[g["scope"] == scope]
            if len(gs):
                rec[scope] = gs["macro_f1"].mean()
                rec[scope + "_std"] = gs["macro_f1"].std(ddof=0)
                per_scope_means.append(gs["macro_f1"].mean())
        if per_scope_means:
            rec["avg_unweighted"] = float(np.mean(per_scope_means))
        rec["n_runs"] = len(g)
        recs.append(rec)
    t = pd.DataFrame(recs)
    t["order"] = t["system"].map({s: i for i, s in enumerate(systems_order)}).fillna(99)
    t = t.sort_values("order").drop(columns="order")
    t.to_csv(out_dir / f"table_{regime}.csv", index=False)
    # LaTeX rows
    lines = []
    for _, r in t.iterrows():
        cells = [str(r["system"])]
        for scope in DOMAINS:
            if scope in r and not pd.isna(r.get(scope)):
                cells.append(_fmt(r[scope], r.get(scope + "_std")))
            else:
                cells.append("--")
        cells.append(f"{100 * r['avg_unweighted']:.1f}" if not pd.isna(r.get("avg_unweighted")) else "--")
        lines.append(" & ".join(cells) + r" \\")
    (out_dir / f"table_{regime}.tex").write_text("\n".join(lines), encoding="utf-8")


def table_ablation_probes(df: pd.DataFrame, out_dir: Path) -> None:
    chain = ["xlmr_ft", "xlmr_dapt", "xlmr_dapt_film_pol", "xlmr_dapt_film", "full"]
    labels = ["XLM-R fine-tune", "+ DAPT", "+ Polarity injection (FiLM)", "+ Morphology cues", "+ SupCon (Full)"]
    d = df[(df["regime"] == "indomain") & (df["tag"] == "") & (df["system"].isin(chain))]
    recs = []
    for sysname, lab in zip(chain, labels):
        g = d[d["system"] == sysname]
        if g.empty:
            continue
        rec = {"variant": lab, "system": sysname,
               "macro_f1_domain_avg": g.groupby("scope")["macro_f1"].mean().mean()}
        for k in ("nn_at_1", "nn_at_5", "nn_at_10", "nn_at_20", "nmi", "ari", "silhouette",
                  "davies_bouldin", "anisotropy", "twonn_id",
                  "centroid_within_class_cross_domain", "centroid_cross_class"):
            if k in g:
                rec[k] = g[k].mean()
        recs.append(rec)
    if recs:
        pd.DataFrame(recs).to_csv(out_dir / "table_ablation_probes.csv", index=False)


def table_sensitivity(df: pd.DataFrame, out_dir: Path) -> None:
    d = df[df["regime"] == "pooled"]
    if d.empty:
        return
    # silver fraction curve
    rows = []
    for (tag, frac, mode), g in d[d["system"] == "full"].groupby(["tag", "silver_frac", "pool_mode"]):
        rows.append({"tag": tag or "main", "silver_frac": frac, "pool_mode": mode,
                     "macro_f1_mean": g["macro_f1"].mean(), "macro_f1_std": g["macro_f1"].std(ddof=0),
                     "neu_f1_mean": g["neu_f1"].mean(), "n_seeds": len(g),
                     "train_total_mean": g["train_total"].mean()})
    pd.DataFrame(rows).sort_values(["pool_mode", "silver_frac"]).to_csv(out_dir / "table_silver_sensitivity.csv", index=False)
    # lambda/tau grid
    lt = d[(d["system"] == "full") & d["tag"].str.startswith("lam")]
    if len(lt):
        g = lt.groupby(["supcon_lambda", "supcon_tau"])["macro_f1"].agg(["mean", "std", "count"]).reset_index()
        g.to_csv(out_dir / "table_lambda_tau.csv", index=False)


def table_lodo_sampler(df: pd.DataFrame, out_dir: Path) -> None:
    d = df[(df["regime"] == "lodo") & df["system"].isin(["full", "full_sampler_random", "full_sampler_indomain"])]
    if d.empty:
        return
    rows = []
    for sysname, g in d.groupby("system"):
        rows.append({"system": sysname,
                     "lodo_avg": g.groupby("scope")["macro_f1"].mean().mean(),
                     "centroid_within": g["centroid_within_class_cross_domain"].mean() if "centroid_within_class_cross_domain" in g else None,
                     "n_runs": len(g)})
    pd.DataFrame(rows).to_csv(out_dir / "table_sampler.csv", index=False)


def table_calibration(df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for (regime, sysname), g in df[df["system"].isin(["xlmr_ft", "full"])].groupby(["regime", "system"]):
        if "ece_before" not in g or g["ece_before"].isna().all():
            continue
        rows.append({"regime": regime, "system": sysname,
                     "ece_before": g["ece_before"].mean(), "ece_after": g["ece_after"].mean(),
                     "mce_before": g["mce_before"].mean(), "mce_after": g["mce_after"].mean(),
                     "brier_before": g["brier_before"].mean(), "brier_after": g["brier_after"].mean(),
                     "temperature": g["temperature"].mean(), "neu_ece_before": g["neu_ece_before"].mean()})
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "table_calibration.csv", index=False)


def geometry_correlation(df: pd.DataFrame, out_dir: Path) -> None:
    """Run-level Spearman correlation between geometry probes and LODO F1
    (Reviewer 2: geometry <-> robustness link, correlational framing)."""
    from scipy.stats import spearmanr

    d = df[(df["regime"] == "lodo") & df["nn_at_10"].notna()]
    if len(d) < 8:
        return
    rows = []
    for k in ("nn_at_10", "nmi", "anisotropy", "centroid_within_class_cross_domain"):
        if k in d and d[k].notna().sum() >= 8:
            rho, p = spearmanr(d[k], d["macro_f1"], nan_policy="omit")
            rows.append({"probe": k, "spearman_rho_vs_lodo_f1": float(rho), "p": float(p), "n_runs": int(d[k].notna().sum())})
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "table_geometry_correlation.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    cfg = load_config()
    setup_logging(Path(cfg.paths.runs_dir) / "logs" / "report.log")
    systems = load_systems()
    out_dir = Path(cfg.paths.runs_dir) / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = collect(cfg)
    if df.empty:
        log.warning("No finished runs found")
        return
    df.to_csv(out_dir / "runs_master.csv", index=False)
    order_in = ["tfidf_linear", "fasttext_mlp", "mbert_ft", "xlmr_ft", "mdeberta_ft", "berturk_ft",
                "allma_ft", "xlmr_dapt", "xlmr_add", "xlmr_film", "full", "full_mdeberta", "full_mbert"]
    table_regime(df, "indomain", out_dir, order_in)
    order_lodo = ["xlmr_ft", "allma_ft", "mdeberta_ft", "xlmr_dapt", "xlmr_dann", "xlmr_film",
                  "xlmr_dapt_film", "full"]
    table_regime(df, "lodo", out_dir, order_lodo)
    table_ablation_probes(df, out_dir)
    table_sensitivity(df, out_dir)
    table_lodo_sampler(df, out_dir)
    table_calibration(df, out_dir)
    geometry_correlation(df, out_dir)
    # copy prepared-data documentation tables into the report folder
    p = prep_dir(cfg)
    for name in ("dedup_report.json", "truncation_report.json", "gold_summary.json", "split_counts.json"):
        src = p / name
        if src.exists():
            (out_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    log.info("Report written to %s (%d runs aggregated)", out_dir, len(df))


if __name__ == "__main__":
    main()
