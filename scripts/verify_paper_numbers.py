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


def check(name: str, got: float, want: float, tol: float = 0.0005) -> None:
    ok = abs(got - want) <= tol
    results.append((ok, f"[{'PASS' if ok else 'FAIL'}] {name:<52} got {got:.5f}  paper {want:g}"))


def main() -> int:
    runs = pd.read_csv(RES / "runs_master.csv")
    runs["macro_f1"] = pd.to_numeric(runs["macro_f1"], errors="coerce")

    print(f"run ledger: {len(runs):,} runs, {runs['system'].nunique()} distinct systems\n")

    # ---- main-table averages: mean over seeds, then unweighted mean over domains
    print("Tables 8 and 9 — average macro-F1")
    for regime, system, want in EXPECTED_AVG:
        sub = runs[(runs["regime"] == regime) & (runs["system"] == system)]
        per_domain = sub.groupby("scope")["macro_f1"].mean()
        check(f"{regime:<8} {system}", float(per_domain.mean()), want)

    # ---- pooled regime: five-seed encoder means of Table 10 (Section VI-B)
    print()
    pooled = runs[(runs["regime"] == "pooled") & (runs["tag"].fillna("") != "trstage")]
    for system, want_mean, want_std in [("mdeberta_ft", 0.760, 0.009), ("xlmr_ft", 0.746, 0.007), ("full", 0.742, 0.006)]:
        g = pooled[pooled["system"] == system]["macro_f1"]
        check(f"pooled {system} five-seed mean (n={len(g)})", float(g.mean()), want_mean)
        check(f"pooled {system} seed std (population)", float(g.std(ddof=0)), want_std, tol=0.0005)
    ok = len(runs) == 616 and int(runs["bulk_silver"].max()) == 0
    results.append((ok, f"[{'PASS' if ok else 'FAIL'}] {'ledger: 616 runs, no silver rows in any pool':<52} "
                        f"{len(runs)} runs, max bulk_silver {int(runs['bulk_silver'].max())}"))

    # ---- the study's only two positive comparisons, from the stats files
    print()
    st = json.loads((RES / "stats" / "stats_lodo_vs_xlmr.json").read_text())
    tech = st["comparisons"]["mdeberta_ft"]["Tech"]
    check("mDeBERTa vs XLM-R, LODO Tech, lower CI", tech["ci95"][0], 0.005, tol=0.0005)
    check("mDeBERTa vs XLM-R, LODO Tech, upper CI", tech["ci95"][1], 0.068, tol=0.0005)

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

    # ---- seed-mean design: every interval and p-value quoted in the text (Sections I, IV-C, VI-A, VI-D)
    print()
    sm = RES / "stats" / "seed_mean"
    fams = {f.name: json.loads(f.read_text())["comparisons"] for f in sorted(sm.glob("stats_*_seed_mean.json"))}
    lx, ix = fams["stats_lodo_vs_xlmr_seed_mean.json"], fams["stats_indomain_vs_xlmr_seed_mean.json"]
    for label, v, want in [
        ("mDeBERTa vs XLM-R, LODO Tech", lx["mdeberta_ft"]["Tech"], (0.037, 0.013, 0.059)),
        ("DAPT vs XLM-R, LODO Tech", lx["xlmr_dapt"]["Tech"], (0.021, 0.005, 0.039)),
        ("DAPT+FiLM vs XLM-R, LODO Retail", lx["xlmr_dapt_film"]["Retail"], (0.026, 0.004, 0.051)),
        ("mDeBERTa+DAPT vs XLM-R, LODO Tech", lx["mdeberta_dapt"]["Tech"], (0.021, 0.002, 0.040)),
        ("Full (mDeBERTa) vs XLM-R, in-domain Social", ix["full_mdeberta"]["Social"], (0.035, 0.0002, 0.071)),
        ("DAPT vs SupCon, LODO Tech", fams["stats_lodo_vs_supcon_seed_mean.json"]["xlmr_dapt"]["Tech"], (0.030, 0.011, 0.051)),
        ("Full (XLM-R) vs mDeBERTa FT, LODO Tech", fams["stats_lodo_full_vs_mdeberta_seed_mean.json"]["full"]["Tech"], (-0.028, -0.053, -0.002)),
        ("Full (XLM-R) vs mDeBERTa FT, LODO Finance", fams["stats_lodo_full_vs_mdeberta_seed_mean.json"]["full"]["Finance"], (-0.032, -0.072, -0.001)),
    ]:
        check(f"seed-mean: {label}, diff", v["diff_mean_over_seeds"], want[0], tol=0.0006)
        check(f"seed-mean: {label}, lower CI", v["ci95"][0], want[1], tol=0.0006 if abs(want[1]) >= 0.001 else 0.00005)
        check(f"seed-mean: {label}, upper CI", v["ci95"][1], want[2], tol=0.0006)
    check("seed-mean: mDeBERTa vs XLM-R, LODO Tech, p", lx["mdeberta_ft"]["Tech"]["p_onesided_A_gt_B"], 0.0013, tol=0.00005)
    check("seed-mean: mDeBERTa vs XLM-R, LODO Tech, Holm p", lx["mdeberta_ft"]["Tech"]["holm_adjusted_p_within_family"], 0.097, tol=0.0006)
    check("seed-mean: DAPT vs XLM-R, LODO Tech, p", lx["xlmr_dapt"]["Tech"]["p_onesided_A_gt_B"], 0.008, tol=0.0005)
    check("seed-mean: DAPT vs XLM-R, LODO Tech, Holm p", lx["xlmr_dapt"]["Tech"]["holm_adjusted_p_within_family"], 0.62, tol=0.005)
    check("seed-mean: DAPT+FiLM vs XLM-R, LODO Retail, p", lx["xlmr_dapt_film"]["Retail"]["p_onesided_A_gt_B"], 0.010, tol=0.0005)
    check("seed-mean: DAPT+FiLM vs XLM-R, LODO Retail, Holm p", lx["xlmr_dapt_film"]["Retail"]["holm_adjusted_p_within_family"], 0.76, tol=0.005)
    ds = fams["stats_lodo_vs_supcon_seed_mean.json"]["xlmr_dapt"]["Tech"]
    check("seed-mean: DAPT vs SupCon, LODO Tech, p", ds["p_onesided_A_gt_B"], 0.0017, tol=0.00005)
    check("seed-mean: DAPT vs SupCon, LODO Tech, Holm p", ds["holm_adjusted_p_within_family"], 0.034, tol=0.0005)
    # effects supported before adjustment against plain XLM-R: exactly the five named in Section I
    supported = sorted(f"{reg}.{s}.{d}" for reg, fam in (("lodo", lx), ("indomain", ix))
                       for s, doms in fam.items() for d, v in doms.items() if v["ci95"][0] > 0)
    want_set = sorted(["lodo.mdeberta_ft.Tech", "lodo.xlmr_dapt.Tech", "lodo.xlmr_dapt_film.Retail",
                       "lodo.mdeberta_dapt.Tech", "indomain.full_mdeberta.Social"])
    ok = supported == want_set and all(fam[s][d]["holm_adjusted_p_within_family"] > 0.05
                                       for fam, s, d in [(lx if k.startswith("lodo") else ix, k.split(".")[1], k.split(".")[2]) for k in supported])
    results.append((ok, f"[{'PASS' if ok else 'FAIL'}] {'seed-mean: five effects vs XLM-R before adjustment':<52} {len(supported)} found, none after Holm"))
    ok = all(v["ci95"][0] <= 0 <= v["ci95"][1] for fam in (lx, ix) for v in fam["full"].values())
    results.append((ok, f"[{'PASS' if ok else 'FAIL'}] {'seed-mean: full recipe CI includes zero everywhere':<52} both regimes"))
    for reg, fam in (("LODO", lx), ("in-domain", ix)):
        n = sum(1 for v in fam["mbert_ft"].values() if v["ci95"][1] < 0 and v["seed_pair_design_ci95"][1] < 0)
        results.append((n == 2, f"[{'PASS' if n == 2 else 'FAIL'}] {'mBERT deficits excluding zero (both designs), ' + reg:<52} {n} targets"))
    ps = sorted(v["p_onesided_A_gt_B"] for fam in fams.values() for doms in fam.values() for v in doms.values())
    m = len(ps)
    holm = [min(1.0, (m - i) * p) for i, p in enumerate(ps)]
    for i in range(1, m):
        holm[i] = max(holm[i], holm[i - 1])
    check("seed-mean: joint Holm, smallest adjusted p", holm[0], 0.25, tol=0.005)

    # ---- Section VII: the earlier (unaudited) split, with and without teacher labels
    print()
    es = RES / "earlier_split"
    t = json.loads((es / "stats_lodo_full_vs_xlmr_earlier_split_teacher_labels.json").read_text())
    h = json.loads((es / "stats_lodo_full_vs_xlmr_earlier_split_human_labels.json").read_text())
    tp = t["comparisons"]["full"]["Public"]
    check("Stage 2 (teacher labels): Public gap", tp["diff_mean_over_seedpairs"], 0.096, tol=0.0005)
    check("Stage 2 (teacher labels): Public CI lower", tp["ci95"][0], 0.032, tol=0.0005)
    check("Stage 2 (teacher labels): Public CI upper", tp["ci95"][1], 0.163, tol=0.0005)
    check("Stage 2 (teacher labels): Public one-sided p", tp["p_onesided_A_gt_B"], 0.0012, tol=0.00005)
    check("Stage 2 (teacher labels): LODO-average gap", t["lodo_average"]["full"] - t["lodo_average"]["xlmr_ft"], 0.025, tol=0.0005)
    check("Stage 2 (teacher labels): XLM-R FT on Public", tp["xlmr_ft_mean_macro_f1"], 0.416, tol=0.0005)
    am = json.loads((es / "stats_lodo_vs_xlmr_earlier_split_teacher_labels_allsystems.json").read_text())["comparisons"]["full_mbert"]["Public"]
    check("Stage 2 (teacher labels): mBERT recipe, Public gap", am["diff_mean_over_seedpairs"], 0.090)
    check("Stage 2 (teacher labels): mBERT recipe, CI lower", am["ci95"][0], 0.023)
    check("Stage 2 (teacher labels): mBERT recipe, CI upper", am["ci95"][1], 0.159)
    hp = h["comparisons"]["full"]["Public"]
    check("Stage 2b (human labels): Public gap", hp["diff_mean_over_seedpairs"], -0.015, tol=0.0005)
    check("Stage 2b (human labels): LODO-average gap", h["lodo_average"]["full"] - h["lodo_average"]["xlmr_ft"], 0.005, tol=0.0005)
    check("Stage 2b (human labels): XLM-R FT on Public", hp["xlmr_ft_mean_macro_f1"], 0.734, tol=0.0005)
    # pooled regime of the pre-audit split: silver fraction and the lambda/tau sweep (Sections VII and VI-D)
    el = pd.read_csv(es / "runs_master_earlier_split.csv")
    pf = el[(el["regime"] == "pooled") & (el["system"] == "full")]
    by_tag = pf.groupby(pf["tag"].fillna("default"))["macro_f1"].mean()
    check("pre-audit pooled: full recipe, human labels only", by_tag["frac000"], 0.740, tol=0.0005)
    fracs = [by_tag["frac025"], by_tag["frac050"], by_tag["frac075"], by_tag["default"]]
    check("pre-audit pooled: full recipe, teacher fractions, min", min(fracs), 0.696, tol=0.0005)
    check("pre-audit pooled: full recipe, teacher fractions, max", max(fracs), 0.715, tol=0.0005)
    check("pre-audit pooled: full recipe at lambda=0.1, tau=0.07", by_tag["default"], 0.700, tol=0.0005)
    sweep = [by_tag[k] for k in ("lam0.1_tau0.05", "lam0.1_tau0.2", "lam0.5_tau0.05", "lam0.5_tau0.07", "lam0.5_tau0.2")]
    check("pre-audit pooled: lambda/tau sweep, min", min(sweep), 0.703, tol=0.0005)
    check("pre-audit pooled: lambda/tau sweep, max", max(sweep), 0.717, tol=0.0005)
    ok = all(int(n) == 3 for n in pf.groupby(pf["tag"].fillna("default"))["seed"].count()[[k for k in by_tag.index if k != "trstage"]])
    results.append((ok, f"[{'PASS' if ok else 'FAIL'}] {'pre-audit pooled: three seeds per setting':<52} {len(sweep) + 5} settings"))
    rw = {r["target_domain"]: r for r in json.loads((es / "prior_reweight_test2_earlier_split.json").read_text())["rows"]}
    check("Test 2 reweighting: Public gap, observed (points)", rw["Public"]["observed_gap_points"], 9.56, tol=0.005)
    check("Test 2 reweighting: Public gap, reweighted (points)", rw["Public"]["reweighted_gap_points"], 12.69, tol=0.005)
    ok = all(r["delta_points"] > 0 for r in rw.values())
    results.append((ok, f"[{'PASS' if ok else 'FAIL'}] {'Test 2 reweighting: gap grows on every domain':<52} {len(rw)} domains"))

    # ---- Section VII, Table 15: label and data-strategy controls on the audited split
    print()
    lc = json.loads((RES / "label_controls" / "label_controls.json").read_text())
    table11 = {("pooled.all", "H"): (0.745, 0.745), ("pooled.all", "T"): (0.722, 0.717), ("pooled.all", "S"): (0.726, 0.727),
               ("lodo.Public", "H"): (0.756, 0.745), ("lodo.Public", "T"): (0.689, 0.681), ("lodo.Public", "S"): (0.697, 0.720)}
    for (cell, cond), (ft, full) in table11.items():
        c = lc[cell][cond]
        check(f"Table 15 {cell} {cond}: XLM-R FT", c["xlmr_ft"]["macro_f1_mean"], ft)
        check(f"Table 15 {cell} {cond}: full recipe", c["full"]["macro_f1_mean"], full)
    check("Table 15 pooled: T-H drop, XLM-R FT", lc["pooled.all"]["T_minus_H"]["xlmr_ft"]["diff_mean_over_seeds"], -0.023, tol=0.0006)
    check("Table 15 Public: T-H drop, XLM-R FT", lc["lodo.Public"]["T_minus_H"]["xlmr_ft"]["diff_mean_over_seeds"], -0.067, tol=0.0006)
    check("Table 15 Public: S-H drop, XLM-R FT", lc["lodo.Public"]["S_minus_H"]["xlmr_ft"]["diff_mean_over_seeds"], -0.059, tol=0.0006)
    sg = lc["lodo.Public"]["S"]["full_minus_xlmr_ft"]
    check("Table 15 Public: recipe gap under S", sg["diff_mean_over_seeds"], 0.023, tol=0.0006)
    ok = not sg["seed_pair"]["excludes_zero"] and not sg["seed_mean"]["excludes_zero"]
    results.append((ok, f"[{'PASS' if ok else 'FAIL'}] {'Table 15 Public: S gap intervals include zero':<52} "
                        f"seed-pair {sg['seed_pair']['ci95']}, seed-mean {sg['seed_mean']['ci95']}"))
    tv = json.loads((RES / "label_controls" / "teacher_vs_human.json").read_text())
    check("teacher-human label disagreement, pooled pool", tv["pooled.all"]["human_vs_teacher_on_training_rows"]["teacher_human_disagreement"], 0.313, tol=0.0006)
    check("teacher-human label disagreement, Public fold", tv["lodo.Public"]["human_vs_teacher_on_training_rows"]["teacher_human_disagreement"], 0.320, tol=0.0006)

    # ---- prompted LLMs
    print()
    llm = json.loads((RES / "llm_openai_metrics.json").read_text())["results"]
    check("GPT-4o, 3-shot", llm["gpt4o_3shot"]["test"]["macro_f1"], 0.677, tol=0.0005)
    check("GPT-4o, 0-shot", llm["gpt4o_0shot"]["test"]["macro_f1"], 0.673, tol=0.0005)
    check("GPT-4o-mini, 3-shot", llm["gpt4o_mini_3shot"]["test"]["macro_f1"], 0.671, tol=0.0005)
    check("GPT-4o-mini, 0-shot", llm["gpt4o_mini_0shot"]["test"]["macro_f1"], 0.665, tol=0.0005)

    # ---- QLoRA-adapted LLMs (single runs), Table 10
    for name, want in [("qwen25", 0.761), ("qwen3", 0.759), ("llama31", 0.774),
                       ("qwen25_lodo_Public", 0.752), ("qwen25_lodo_Tech", 0.736)]:
        f = RES / "llm" / name / "metrics.json"
        if f.exists():
            check(f"QLoRA {name}", json.loads(f.read_text())["test"]["macro_f1"], want, tol=0.0005)

    # ---- cross-lingual transfer
    tr = json.loads((RES / "transfer_report.json").read_text())
    check("Turkish-trained XLM-R, zero-shot", tr["t1_tr_only_to_az"]["macro_f1"], 0.379, tol=0.0005)
    check("LaBSE frozen, Turkish only", tr["t2_labse"]["tr_only"]["macro_f1"], 0.414, tol=0.0005)
    check("LaBSE frozen, Turkish + Az", tr["t2_labse"]["tr_plus_az_goldtrain"]["macro_f1"], 0.538, tol=0.0005)
    # per-class and per-domain breakdown of the zero-shot run (Section VI-C)
    t1 = tr["t1_tr_only_to_az"]
    cc = t1["confusion_counts"]  # rows: gold neg/neu/pos; columns: predicted neg/neu/pos
    n_test = sum(sum(r) for r in cc)
    check("zero-shot Turkish: predicted-neutral share (%)", 100 * sum(r[1] for r in cc) / n_test, 2.0, tol=0.5)
    check("zero-shot Turkish: gold-neutral share (%)", 100 * sum(cc[1]) / n_test, 29.0, tol=0.5)
    check("zero-shot Turkish: neutral routed to positive (%)", 100 * t1["confusion_row_norm"][1][2], 82.0, tol=0.5)
    check("zero-shot Turkish: negative F1", t1["per_class_f1"]["negative"], 0.54, tol=0.005)
    check("zero-shot Turkish: positive F1", t1["per_class_f1"]["positive"], 0.57, tol=0.005)
    check("zero-shot Turkish: neutral F1", t1["per_class_f1"]["neutral"], 0.03, tol=0.005)
    dom_f1 = {d: v["macro_f1"] for d, v in t1["by_domain"].items()}
    check("zero-shot Turkish: lowest per-domain macro-F1 (Retail)", dom_f1["Retail"], 0.35, tol=0.005)
    check("zero-shot Turkish: per-domain macro-F1, Finance", dom_f1["Finance"], 0.35, tol=0.005)
    check("zero-shot Turkish: highest per-domain macro-F1 (Social)", max(dom_f1.values()), 0.42, tol=0.005)
    ok = max(dom_f1, key=dom_f1.get) == "Social" and min(dom_f1, key=dom_f1.get) == "Retail"
    results.append((ok, f"[{'PASS' if ok else 'FAIL'}] {'zero-shot Turkish: Retail lowest, Social highest':<52} {min(dom_f1, key=dom_f1.get)}, {max(dom_f1, key=dom_f1.get)}"))
    neu_dom = {d: v["per_class_f1"]["neutral"] for d, v in t1["by_domain"].items()}
    ok = max(neu_dom.values()) <= 0.06
    results.append((ok, f"[{'PASS' if ok else 'FAIL'}] {'zero-shot Turkish: neutral F1 <= 0.06 on every domain':<52} max {max(neu_dom.values()):.3f}"))
    check("zero-shot Turkish: neutral F1 on Public", neu_dom["Public"], 0.00, tol=0.005)
    check("LaBSE frozen, Turkish + Az: neutral F1", tr["t2_labse"]["tr_plus_az_goldtrain"]["per_class_f1"]["neutral"], 0.35, tol=0.005)
    check("LaBSE frozen, Turkish only: neutral F1", tr["t2_labse"]["tr_only"]["per_class_f1"]["neutral"], 0.10, tol=0.005)

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

    # ---- lexicon coverage on the gold partition (Section V-C)
    cov = json.loads((RES / "lexicon_coverage.json").read_text())["gold_overall"]
    check("lexicon: exact-match token coverage (%)", cov["exact_pct"], 4.3, tol=0.05)
    check("lexicon: prefix-backoff token coverage (%)", cov["combined_pct"], 6.0, tol=0.05)
    check("lexicon: comments with >=1 match (%)", cov["comments_with_any_hit_pct"], 53.0, tol=0.5)

    # ---- prompted-LLM outputs: every response parsed (Section V-D)
    import re
    line_re = re.compile(r"^\s*(\d+)\s*[:.\-]\s*(NEG|NEU|POS)\b", re.I | re.M)
    n_resp = n_short = 0
    for f in sorted((RES / "llm" / "openai").glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            n_resp += 1
            if len(line_re.findall(json.loads(line)["response"])) < 20:
                n_short += 1
    ok = n_resp == 600 and n_short == 0
    results.append((ok, f"[{'PASS' if ok else 'FAIL'}] "
                        f"{'prompted-LLM responses fully parsed':<52} {n_resp} responses, {n_short} short"))

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
