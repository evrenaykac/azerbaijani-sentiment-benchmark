"""Run every check that, when skipped, cost this project a day of compute.

Each check below corresponds to a concrete failure from the first full run:
a half-precision checkpoint that trained to a constant class while still
producing a plausible-looking score; a gated model that failed thirty minutes
into a queue; an external dataset whose column names silently did not match; a
feature cache that served stale vectors because its key ignored the code that
built them. None of these announced themselves - they all looked like results.

Exit code 0 only when everything the queue will need is present and verified.

    python3 -m azsent.preflight              # full check
    python3 -m azsent.preflight --skip-net   # cached assets only, no network
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

RED, YEL, GRN, RST = "\033[31m", "\033[33m", "\033[32m", "\033[0m"
if not sys.stdout.isatty():
    RED = YEL = GRN = RST = ""

results: list[tuple[str, str, str]] = []


def ok(name, detail=""):    results.append(("OK", name, detail))
def warn(name, detail=""):  results.append(("WARN", name, detail))
def bad(name, detail=""):   results.append(("FAIL", name, detail))


# --------------------------------------------------------------------------- env
def check_env(cfg) -> None:
    import torch

    ok("python", sys.version.split()[0])
    try:
        import transformers, datasets
        ok("packages", "torch %s | transformers %s | datasets %s"
           % (torch.__version__, transformers.__version__, datasets.__version__))
    except Exception as e:  # noqa: BLE001
        bad("packages", str(e))
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n:
        names = {torch.cuda.get_device_name(i) for i in range(n)}
        free = min(torch.cuda.mem_get_info(i)[0] for i in range(n)) / 2**30
        ok("gpu", "%d device(s) (%s), lowest free memory %.1f GB" % (n, ", ".join(names), free))
        if free < 10:
            warn("gpu memory", "%.1f GB free - micro_batch may need to be lowered" % free)
    else:
        bad("gpu", "CUDA not visible")
    runs = Path(cfg.paths.runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    gb = shutil.disk_usage(runs).free / 2**30
    (ok if gb > 80 else bad)("disk", "%s: %.0f GB free (the QLoRA models need ~50 GB)" % (runs, gb))


# ------------------------------------------------------------------------- data
def check_data(cfg) -> "object | None":
    import pandas as pd

    src = Path(cfg.paths.data_dir) / cfg.paths.dataset_name
    if not src.exists():
        bad("data file", "not found: %s" % src)
        return None
    df = (pd.read_parquet(src) if src.suffix == ".parquet"
          else pd.read_excel(src, sheet_name=cfg.paths.get("dataset_sheet", "data")))
    ok("data file", "%s (%d rows)" % (src.name, len(df)))

    dc = cfg.get("data_contract", {})
    miss = [c for c in dc.get("require_columns", []) if c not in df.columns]
    (bad if miss else ok)("schema", "missing columns: %s" % miss if miss else "all required columns present")
    if miss:
        return None

    doms = sorted(set(df["domain"].dropna()))
    canon = sorted(cfg.corpus.domains.keys())
    (ok if doms == canon else bad)("domain names", "%s" % doms if doms == canon else
                                   "not canonical: %s (expected %s)" % (doms, canon))
    labs = sorted(set(df["label"].dropna()))
    (ok if labs == sorted(cfg.corpus.labels) else bad)("labels", str(labs))
    (ok if df["uid"].is_unique else bad)("unique uid", "")

    from .normalize import dedup_key
    key = df["text"].astype(str).map(dedup_key)
    dup = int(key.duplicated().sum())
    (ok if dup == 0 else bad)("unique text", "duplicates: %d" % dup)

    grp = df["video_id_hash"].where(df["video_id_hash"].notna(), "solo_" + df["uid"].astype(str))
    spans = int(df.assign(_g=grp).groupby("_g")["split"].nunique().gt(1).sum())
    (ok if spans == 0 else bad)("group integrity", "groups spanning more than one split: %d" % spans)

    ev = df["gold_role"].isin(["gold_dev", "gold_test"])
    tr = df["split"] == "train"
    t_leak = len(set(key[ev]) & set(key[tr]))
    g_leak = len(set(grp[ev]) & set(grp[tr]))
    (ok if t_leak == 0 else bad)("text leakage", "evaluation texts in training: %d" % t_leak)
    (ok if g_leak == 0 else bad)("group leakage", "evaluation groups in training: %d" % g_leak)

    gold = df[df["gold_role"].notna()]
    tol = float(dc.get("gold_prior_tolerance_pct", 1.5))
    pri = {r: (g["label"].value_counts(normalize=True) * 100)
           for r, g in gold.groupby("gold_role")}
    worst, where = 0.0, ""
    for lab in cfg.corpus.labels:
        vals = {r: float(p.get(lab, 0)) for r, p in pri.items()}
        d = max(vals.values()) - min(vals.values())
        if d > worst:
            worst, where = d, lab
    (ok if worst <= tol else bad)(
        "label prior", "largest gap between partitions %.2f points (%s), tolerance %.1f" % (worst, where, tol))

    tol_d = float(dc.get("domain_balance_tolerance_pct", 2.0))
    dpr = {r: (g["domain"].value_counts(normalize=True) * 100) for r, g in gold.groupby("gold_role")}
    worst_d, where_d = 0.0, ""
    for dom in canon:
        vals = {r: float(p.get(dom, 0)) for r, p in dpr.items()}
        d = max(vals.values()) - min(vals.values())
        if d > worst_d:
            worst_d, where_d = d, dom
    (ok if worst_d <= tol_d else bad)(
        "domain balance", "largest gap between partitions %.2f points (%s), tolerance %.1f" % (worst_d, where_d, tol_d))

    cov = 100 * float(df["video_id_hash"].notna().mean())
    (ok if cov >= 50 else warn)("source-group id coverage", "%.1f%% (grouping applies to these rows)" % cov)
    _check_annotators(df, cfg)
    return df


def _fleiss_kappa(counts) -> float:
    import numpy as np

    n, N = counts.shape[0], counts.sum(1)[0]
    p = counts.sum(0) / (n * N)
    P = ((counts ** 2).sum(1) - N) / (N * (N - 1))
    denom = 1 - (p ** 2).sum()
    return float((P.mean() - (p ** 2).sum()) / denom) if denom > 0 else float("nan")


def _check_annotators(df, cfg=None) -> None:
    """Report real inter-annotator agreement - and refuse to bless fake agreement.

    Columns that were derived from the final label rather than collected
    independently produce a kappa near 1.0 with essentially no disagreements.
    Published as an agreement table that is not a weak result, it is a
    fabricated measurement, so this check fails rather than warns.
    """
    import numpy as np

    cols = [c for c in df.columns
            if c.lower().startswith("ann") and not c.lower().startswith("annotator_id")]
    if len(cols) < 2:
        warn("annotator columns", "absent - inter-annotator agreement (kappa) cannot be computed")
        return
    sub = df[cols + ["label"]].dropna()
    if len(sub) < 50:
        warn("annotator columns", "filled on %d rows - too few for an agreement estimate" % len(sub))
        return

    # An adjudicator column is the final decision, not an independent annotation:
    # including it would both inflate agreement and trip the fabrication check
    # below, since it equals the final label by construction. Named in the config
    # when known; otherwise the column that reproduces the final label almost
    # exactly is treated as the adjudicator - but only one, so a set derived
    # wholesale from the label still fails.
    dc = (cfg or {}).get("data_contract", {}) if hasattr(cfg, "get") else {}
    adjudicator = str(dc.get("adjudicator_column", "") or "")
    match_all = {c: float((sub[c] == sub["label"]).mean()) for c in cols}
    if adjudicator not in cols:
        top = max(match_all, key=match_all.get)
        adjudicator = top if match_all[top] > 0.995 and len(cols) > 2 else ""
    per_ann = [c for c in cols if c != adjudicator]
    if len(per_ann) < 2:
        warn("annotator columns", "only one independent annotator column - kappa cannot be computed")
        return

    labels = sorted(set(np.ravel(sub[per_ann].values)))
    counts = np.stack([(sub[per_ann].values == c).sum(1) for c in labels], axis=1).astype(float)
    kappa = _fleiss_kappa(counts)
    disagree = float((counts.max(1) < len(per_ann)).mean())
    worst = max(match_all[c] for c in per_ann)

    msg = ("%s | n=%d | Fleiss kappa %.4f | disagreement %.2f%% | "
           "highest overlap with the final label %.2f%%"
           % (", ".join(per_ann), len(sub), kappa, disagree * 100, worst * 100))
    if adjudicator:
        msg += " | adjudicator: %s (EXCLUDED from agreement, %.2f%% identical to final)" % (
            adjudicator, match_all[adjudicator] * 100)
    if kappa > 0.95 or disagree < 0.02 or worst > 0.995:
        bad("annotator columns", msg +
            " -- these columns do not look independently collected (they may be derived "
            "from the final label). Do NOT publish a kappa table from them.")
    elif kappa < 0.4:
        warn("annotator columns", msg + " -- low agreement; report the labeling protocol")
    else:
        ok("annotator columns", msg)


# ---------------------------------------------------------------------- lexicon
def check_lexicon(cfg, df) -> None:
    from .lexicon import load_lexicon
    from .lexcov import coverage

    try:
        lex = load_lexicon(cfg.paths.data_dir, cfg.paths.lexicon_subdir)
    except Exception as e:  # noqa: BLE001
        bad("lexicon", str(e))
        return
    uni = {k for k in lex if " " not in k}
    if df is None:
        ok("lexicon", "%d single-word entries" % len(uni))
        return
    minp = int(cfg.get("lexicon", {}).get("min_prefix_len", 4))
    # random sample, not the first 20k rows: the export is grouped by domain, so
    # a head slice measures one domain and silently disagrees with lexcov's number
    from .normalize import normalize_text

    txt = df["text"].astype(str)
    txt = txt.sample(n=20000, random_state=42) if len(txt) > 20000 else txt
    # normalize first: the lexicon matches normalized tokens, so measuring on raw
    # text understates coverage and disagrees with lexcov's reported number
    c = coverage([normalize_text(t) for t in txt.tolist()], uni, minp)
    msg = ("%d single-word entries | corpus sample (normalized): exact match %.2f%% | with prefix back-off %.2f%% | comments with at least one match %.1f%%"
           % (len(uni), c["exact_pct"], c["combined_pct"], c["comments_with_any_hit_pct"]))
    (ok if c["combined_pct"] >= 5 else warn)("lexicon coverage", msg)


# ----------------------------------------------------------------------- models
def check_models(cfg, skip_net: bool) -> None:
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    cache = str(Path(cfg.paths.runs_dir) / "hf_cache")
    if skip_net:
        os.environ["HF_HUB_OFFLINE"] = "1"
    for key, spec in cfg.backbones.items():
        cands = spec if isinstance(spec, (list, tuple)) else [spec]
        got = None
        for c in cands:
            try:
                AutoConfig.from_pretrained(c, cache_dir=cache)
                got = c
                break
            except Exception:  # noqa: BLE001
                continue
        if not got:
            bad("model %s" % key, "no candidate could be resolved: %s" % list(cands))
            continue
        try:
            m = AutoModel.from_pretrained(got, cache_dir=cache)
            dt = next(m.parameters()).dtype
            del m
            AutoTokenizer.from_pretrained(got, cache_dir=cache)
            forced = [str(x).lower() for x in cfg.train.get("force_fp32_backbones", [])]
            if dt != torch.float32 and key.lower() not in forced:
                bad("model %s" % key, "%s -> %s: checkpoint is not fp32 and the backbone is not in train.force_fp32_backbones"
                    % (got, dt))
            else:
                ok("model %s" % key, "%s (%s)" % (got, dt))
        except Exception as e:  # noqa: BLE001
            bad("model %s" % key, "%s: %s" % (got, str(e)[:120]))

    for name, mid in cfg.llm.qlora_models.items():
        try:
            AutoTokenizer.from_pretrained(mid, cache_dir=cache)
            ok("llm %s" % name, mid)
        except Exception as e:  # noqa: BLE001
            bad("llm %s" % name, "%s: %s" % (mid, str(e)[:120]))
    if not os.environ.get("OPENAI_API_KEY"):
        warn("openai", "OPENAI_API_KEY not set - the prompted GPT-4o rows will be skipped")


# --------------------------------------------------------------------- datasets
def check_transfer(cfg) -> None:
    from datasets import load_dataset

    from .transfer import _LABEL_CANDS, _TEXT_CANDS

    ids = [cfg.transfer.turkish.trsav1, cfg.transfer.turkish.winvoker, cfg.transfer.kazakh,
           cfg.transfer.get("uzbek_hf"), cfg.transfer.azerbaijani_external]
    for ds_id in [i for i in ids if i]:
        try:
            d = load_dataset(ds_id)
            sp = next(iter(d))
            cols = list(d[sp].column_names)
            low = {c.lower() for c in cols}
            t = next((c for c in _TEXT_CANDS if c.lower() in low), None)
            l = next((c for c in _LABEL_CANDS if c.lower() in low), None)
            if t and l:
                ok("dataset %s" % ds_id.split("/")[-1], "text=%s label=%s" % (t, l))
            else:
                bad("dataset %s" % ds_id.split("/")[-1],
                    "columns not matched: %s (text=%s label=%s)" % (cols, t, l))
        except Exception as e:  # noqa: BLE001
            bad("dataset %s" % ds_id.split("/")[-1], str(e)[:140])


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-net", action="store_true", help="check cached assets only")
    ap.add_argument("--skip-transfer", action="store_true")
    ap.add_argument("--json", default=None, help="also write the results to this file")
    a = ap.parse_args()

    from .config import load_config
    cfg = load_config()

    check_env(cfg)
    df = check_data(cfg)
    check_lexicon(cfg, df)
    check_models(cfg, a.skip_net)
    if not a.skip_transfer:
        check_transfer(cfg)

    width = max(len(n) for _, n, _ in results) + 2
    print()
    for status, name, detail in results:
        col = {"OK": GRN, "WARN": YEL, "FAIL": RED}[status]
        print("%s%-6s%s %-*s %s" % (col, status, RST, width, name, detail))
    n_bad = sum(1 for s, _, _ in results if s == "FAIL")
    n_warn = sum(1 for s, _, _ in results if s == "WARN")
    print("\n%d checks | %s%d failed%s | %s%d warnings%s"
          % (len(results), RED if n_bad else "", n_bad, RST, YEL if n_warn else "", n_warn, RST))
    if a.json:
        Path(a.json).write_text(json.dumps(
            [{"status": s, "check": n, "detail": d} for s, n, d in results],
            ensure_ascii=False, indent=2), encoding="utf-8")
    if n_bad:
        print("\n%sDo not start the runs.%s Fix the failures above first." % (RED, RST))
        sys.exit(1)
    print("\n%sReady.%s The runs can start." % (GRN, RST))


if __name__ == "__main__":
    main()
