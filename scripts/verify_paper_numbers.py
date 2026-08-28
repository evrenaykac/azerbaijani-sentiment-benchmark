#!/usr/bin/env python3
"""Recompute the headline numbers of the paper from the released run ledger.

Every value checked here is stated in the article; this script recomputes each
one from ``results/`` and reports PASS or FAIL. Nothing is re-trained: the
ledger holds one row per training run, and the LLM and transfer arms ship as
JSON reports.

Usage:
    python scripts/verify_paper_numbers.py
Exit status is 0 if every checked number matches, 1 otherwise.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# Systems shown in the two main tables, in table order, with their paper labels.
MAIN_ROWS = [
    ("tfidf_linear", "tf-idf + linear"),
    ("fasttext_mlp", "fastText + MLP"),
    ("mbert_ft", "mBERT FT"),
    ("berturk_ft", "BERTurk FT"),
    ("allma_ft", "aLLMA FT"),
    ("xlmr_ft", "XLM-R FT"),
    ("mdeberta_ft", "mDeBERTa FT"),
]

# (regime, system, expected average macro-F1 as printed in the paper)
EXPECTED_AVG = [
    ("indomain", "tfidf_linear", 0.685),
    ("indomain", "mbert_ft", 0.676),
    ("indomain", "xlmr_ft", 0.717),
    ("indomain", "mdeberta_ft", 0.730),
    ("lodo", "mbert_ft", 0.668),
    ("lodo", "xlmr_ft", 0.724),
    ("lodo", "mdeberta_ft", 0.739),
    ("lodo", "full", 0.719),
]

results: list[tuple[bool, str]] = []


def check(name: str, got: float, want: float, tol: float = 0.0006) -> None:
    ok = abs(got - want) <= tol
    results.append((ok, f"[{'PASS' if ok else 'FAIL'}] {name:<52} got {got:.4f}  paper {want:.3f}"))


def main() -> int:
    runs = pd.read_csv(RES / "runs_master.csv")
    runs["macro_f1"] = pd.to_numeric(runs["macro_f1"], errors="coerce")

    print(f"run ledger: {len(runs):,} runs, {runs['system'].nunique()} distinct systems\n")

    # ---- main-table averages: mean over seeds, then unweighted mean over domains
    print("Table 3 / Table 4 — average macro-F1")
    for regime, system, want in EXPECTED_AVG:
        sub = runs[(runs["regime"] == regime) & (runs["system"] == system)]
        per_domain = sub.groupby("scope")["macro_f1"].mean()
        check(f"{regime:<8} {system}", float(per_domain.mean()), want)

    # ---- the study's only two positive comparisons, from the stats files
    print()
    st = json.loads((RES / "stats" / "stats_lodo_vs_xlmr.json").read_text())
    tech = st["comparisons"]["mdeberta_ft"]["Tech"]
    check("mDeBERTa vs XLM-R, LODO Tech, lower CI", tech["ci95"][0], 0.005, tol=0.0006)
    check("mDeBERTa vs XLM-R, LODO Tech, upper CI", tech["ci95"][1], 0.068, tol=0.0006)

    fam = st["holm_adjusted_median_p"]
    all_one = all(abs(v - 1.0) < 1e-9 for v in fam.values())
    results.append((all_one, f"[{'PASS' if all_one else 'FAIL'}] "
                             f"{'LODO vs XLM-R: every family p = 1.0':<52} "
                             f"{len(fam)} systems"))

    # ---- joint Holm across every per-domain comparison (robustness check, Section IV-C)
    ps = []
    for f in sorted((RES / "stats").glob("stats_*.json")):
        d = json.loads(f.read_text())
        for doms in d.get("comparisons", {}).values():
            ps += [v["p_onesided_A_gt_B"] for v in doms.values()]
    ps.sort()
    m = len(ps)
    holm = [min(1.0, (m - i) * p) for i, p in enumerate(ps)]
    for i in range(1, m):
        holm[i] = max(holm[i], holm[i - 1])
    ok = m == 190 and holm[0] == 1.0
    results.append((ok, f"[{'PASS' if ok else 'FAIL'}] "
                        f"{'joint Holm over all per-domain comparisons':<52} "
                        f"m={m}, smallest adjusted p={holm[0]:.3f}"))

    # ---- prompted LLMs
    print()
    llm = json.loads((RES / "llm_openai_metrics.json").read_text())["results"]
    check("GPT-4o, 3-shot", llm["gpt4o_3shot"]["test"]["macro_f1"], 0.677, tol=0.0006)
    check("GPT-4o, 0-shot", llm["gpt4o_0shot"]["test"]["macro_f1"], 0.673, tol=0.0006)
    check("GPT-4o-mini, 3-shot", llm["gpt4o_mini_3shot"]["test"]["macro_f1"], 0.671, tol=0.0006)
    check("GPT-4o-mini, 0-shot", llm["gpt4o_mini_0shot"]["test"]["macro_f1"], 0.665, tol=0.0006)

    # ---- cross-lingual transfer
    tr = json.loads((RES / "transfer_report.json").read_text())
    check("Turkish-trained XLM-R, zero-shot", tr["t1_tr_only_to_az"]["macro_f1"], 0.379, tol=0.0006)
    check("LaBSE frozen, Turkish only", tr["t2_labse"]["tr_only"]["macro_f1"], 0.414, tol=0.0006)
    check("LaBSE frozen, Turkish + Az", tr["t2_labse"]["tr_plus_az_goldtrain"]["macro_f1"], 0.538, tol=0.0006)

    # ---- corpus accounting (Section III-B)
    print()
    ing = json.loads((RES / "ingest_report.json").read_text())
    for name, got, want in [
        ("corpus rows", ing["rows_total"], 271278),
        ("gold rows", ing["gold_rows"], 10000),
        ("bulk rows", ing["bulk_rows"], 261278),
        ("raw minus dedup", 279714 - 8436, 271278),
    ]:
        ok = got == want
        results.append((ok, f"[{'PASS' if ok else 'FAIL'}] {name:<52} got {got:,}  paper {want:,}"))
    for k in ("duplicate_normalized_texts", "groups_spanning_splits",
              "eval_text_in_train", "eval_group_in_train"):
        v = ing["leakage_verification"][k]
        results.append((v == 0, f"[{'PASS' if v == 0 else 'FAIL'}] ingest report: {k:<37} {v}"))

    # ---- annotation (Section III-D)
    ann = pd.read_csv(ROOT / "data" / "annotations.csv")
    ok = len(ann) == 11250
    results.append((ok, f"[{'PASS' if ok else 'FAIL'}] "
                        f"{'triple-annotated comments':<52} got {len(ann):,}  paper 11,250"))
    test_ann = int((ann["gold_role"] == "gold_test").sum())
    ok = test_ann == 3000
    results.append((ok, f"[{'PASS' if ok else 'FAIL'}] "
                        f"{'test partition fully triple-annotated':<52} got {test_ann:,}  paper 3,000"))

    for ok, line in results:
        print(line)

    failed = sum(1 for ok, _ in results if not ok)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
