"""Step 2: Merge gold benchmark parts and build the grouped, stratified
5,000/2,000/3,000 gold split.

Accepts any mix of .csv/.xlsx part files in <data_dir>/gold_parts/. Column
names are auto-detected; expected roles:
  text        (yorum | text | comment)
  domain      (domain)
  final label (label | final_label | sentiment_etiketi | adjudicated)
  annotators  (annotator1..3 | ann1..3 | a1..3)   [optional, enables kappa]
  video/group (video_id | video_id_hash | group)  [optional]
  uid         (uid)                                [optional]

Reviewer coverage:
  * R1-2: gold split is grouped by source id (no video/thread spans partitions)
  * R1-4: agreement metrics recomputed from raw annotator columns when present
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, prep_dir
from .data_prep import DOM_CANON
from .normalize import dedup_key, normalize_text
from .utils import log, setup_logging, stable_uid, write_json

LABELS = ["negative", "neutral", "positive"]
LABEL_ALIASES = {
    "neg": "negative", "negative": "negative", "-1": "negative", "-1.0": "negative",
    "neu": "neutral", "neutral": "neutral", "notr": "neutral", "0": "neutral", "0.0": "neutral",
    "pos": "positive", "positive": "positive", "1": "positive", "1.0": "positive",
}

_TEXT_CANDS = ["yorum", "text", "comment", "sentence"]
_LABEL_CANDS = ["final_label", "adjudicated_label", "adjudicated", "label", "sentiment_etiketi", "gold_label"]
_GROUP_CANDS = ["video_id_hash", "video_id", "group_id", "group", "source_id", "thread_id"]
_ANN_PREFIXES = ["annotator", "ann", "a"]


def _norm_label(v) -> str | None:
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    return LABEL_ALIASES.get(s)


def _find_col(df: pd.DataFrame, cands: list[str]) -> str | None:
    low = {c.lower().strip(): c for c in df.columns}
    for c in cands:
        if c in low:
            return low[c]
    return None


def _find_annotator_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    low = {c.lower().strip(): c for c in df.columns}
    for p in _ANN_PREFIXES:
        found = [low[k] for k in low if k.startswith(p) and k[len(p):].strip("_ ").isdigit()]
        if len(found) >= 2:
            return sorted(found)
    return cols


def load_gold_parts(cfg) -> pd.DataFrame:
    gp = Path(cfg.paths.data_dir) / cfg.paths.gold_parts_subdir
    files = sorted([p for p in gp.glob("*") if p.suffix.lower() in (".csv", ".xlsx", ".tsv")]) if gp.exists() else []
    if not files:
        raise FileNotFoundError(
            f"No gold part files found in {gp}. Put the gold benchmark part files "
            f"(csv/xlsx) into this folder, e.g. part01_gold_test.csv."
        )
    frames = []
    for f in files:
        if f.suffix.lower() == ".xlsx":
            d = pd.read_excel(f)
        else:
            d = pd.read_csv(f, sep="\t" if f.suffix.lower() == ".tsv" else ",")
        tcol = _find_col(d, _TEXT_CANDS)
        lcol = _find_col(d, _LABEL_CANDS)
        if tcol is None or lcol is None:
            raise ValueError(f"{f.name}: could not find text/label columns (found: {list(d.columns)})")
        gcol = _find_col(d, _GROUP_CANDS)
        dcol = _find_col(d, ["domain"])
        anns = _find_annotator_cols(d)
        part = pd.DataFrame({
            "text_raw": d[tcol].astype(str),
            "final_label": d[lcol].map(_norm_label),
            "domain": d[dcol] if dcol else None,
            "group_raw": d[gcol].astype(str) if gcol else None,
            "part_file": f.name,
        })
        # --- fixed split detection: filename hint or explicit split column ----
        _SPLIT_ALIASES = {"gold_test": "gold_test", "gold_train": "gold_train", "gold_dev": "gold_dev",
                          "test": "gold_test", "train": "gold_train", "dev": "gold_dev",
                          "validation": "gold_dev", "val": "gold_dev"}
        scol = _find_col(d, ["fixed_split", "split"])
        if scol is not None:
            part["fixed_split"] = d[scol].map(lambda v: _SPLIT_ALIASES.get(str(v).strip().lower()))
        else:
            fixed = next((s for s in ("gold_test", "gold_train", "gold_dev") if s in f.name.lower()), None)
            part["fixed_split"] = fixed
        for i, ac in enumerate(anns[:3], start=1):
            part[f"ann{i}"] = d[ac].map(_norm_label)
        frames.append(part)
        log.info("Gold part %s: %d rows (fixed split: %s, annotator cols: %s)",
                 f.name, len(part), part["fixed_split"].dropna().unique().tolist() or "none", anns or "none")
    g = pd.concat(frames, ignore_index=True)

    # domain canonicalization: allow long names, short names and common aliases
    aliases = dict(DOM_CANON)
    aliases.update({
        "Tech/Digital": "Tech", "Tech": "Tech", "Technology": "Tech",
        "Finance": "Finance", "Social": "Social", "Retail": "Retail", "Public": "Public",
    })
    g["domain"] = g["domain"].map(lambda v: aliases.get(str(v).strip(), str(v).strip()))
    bad_dom = sorted(set(g["domain"]) - set(DOM_CANON.values()))
    if bad_dom:
        raise ValueError(f"Unrecognized gold domains: {bad_dom}")
    g = g[g["final_label"].notna()].copy()
    g["text"] = g["text_raw"].map(normalize_text)
    g["key"] = g["text_raw"].map(dedup_key)
    g["uid"] = ["gold_" + stable_uid(k, str(i)) for i, k in enumerate(g["key"])]
    before = len(g)
    _p = {"gold_test": 3, "gold_dev": 2, "gold_train": 1}
    g["_fprio"] = g.get("fixed_split", pd.Series([None] * len(g))).map(_p).fillna(0)
    g = g.sort_values("_fprio", ascending=False)  # fixed rows win duplicate ties
    g = g.drop_duplicates(subset=["key"]).drop(columns=["_fprio"]).reset_index(drop=True)
    if before - len(g):
        log.info("Gold: dropped %d duplicate texts across parts", before - len(g))
    g["group_id"] = np.where(g["group_raw"].notna(), "gvid_" + g["group_raw"].fillna(""), "gsolo_" + g["uid"])
    return g


def agreement_metrics(g: pd.DataFrame) -> dict:
    """Fleiss' kappa (overall + per-class), mean pairwise Cohen's kappa,
    Krippendorff's alpha (nominal) from ann1..3 columns."""
    ann_cols = [c for c in g.columns if c.startswith("ann")]
    if len(ann_cols) < 2:
        return {"available": False, "reason": "no annotator columns in gold parts"}
    sub = g.dropna(subset=ann_cols)
    if len(sub) == 0:
        return {"available": False, "reason": "annotator columns empty"}
    lab2i = {l: i for i, l in enumerate(LABELS)}
    M = np.zeros((len(sub), 3), dtype=float)  # rating counts per item
    for c in ann_cols:
        for r, l in enumerate(sub[c]):
            M[r, lab2i[l]] += 1
    n_r = M.sum(axis=1)

    def fleiss(Mm):
        nr = Mm.sum(axis=1)
        if (nr < 2).any():
            keep = nr >= 2
            Mm, nr = Mm[keep], nr[keep]
        P_i = ((Mm ** 2).sum(axis=1) - nr) / (nr * (nr - 1))
        p_j = Mm.sum(axis=0) / Mm.sum()
        Pbar, Pe = P_i.mean(), (p_j ** 2).sum()
        return float((Pbar - Pe) / (1 - Pe)) if Pe < 1 else 1.0

    out = {"available": True, "n": int(len(sub)), "n_annotators": len(ann_cols),
           "fleiss_kappa_overall": fleiss(M)}
    # per-class: binary presence-of-class Fleiss
    for j, lab in enumerate(LABELS):
        Mb = np.stack([M[:, j], n_r - M[:, j]], axis=1)
        out[f"fleiss_kappa_{lab}"] = fleiss(Mb)
    # pairwise Cohen
    from sklearn.metrics import cohen_kappa_score
    pk = []
    for i in range(len(ann_cols)):
        for j in range(i + 1, len(ann_cols)):
            pk.append(cohen_kappa_score(sub[ann_cols[i]], sub[ann_cols[j]]))
    out["cohen_kappa_pairwise_mean"] = float(np.mean(pk))
    # Krippendorff alpha (nominal) via coincidence matrix
    vals = sub[ann_cols].to_numpy()
    coin = np.zeros((3, 3))
    for row in vals:
        rs = [lab2i[x] for x in row if x in lab2i]
        m = len(rs)
        if m < 2:
            continue
        for a in rs:
            for b in rs:
                if a != b or rs.count(a) > 1:
                    pass
        for ii in range(m):
            for jj in range(m):
                if ii != jj:
                    coin[rs[ii], rs[jj]] += 1.0 / (m - 1)
    n_tot = coin.sum()
    nc = coin.sum(axis=0)
    Do = n_tot - np.trace(coin)
    De = (n_tot - (nc ** 2).sum() / n_tot)
    out["krippendorff_alpha"] = float(1 - Do / De) if De > 0 else 1.0
    return out


def grouped_stratified_split(g: pd.DataFrame, sizes: dict[str, int], seed: int) -> pd.Series:
    """Assign whole groups to train/dev/test targeting stratification over
    domain x label. Greedy longest-first assignment against remaining quota."""
    rng = np.random.RandomState(seed)
    total = len(g)
    parts = list(sizes.keys())
    quota = {p: {} for p in parts}
    for (dom, lab), cnt in g.groupby(["domain", "final_label"]).size().items():
        for p in parts:
            quota[p][(dom, lab)] = cnt * sizes[p] / total
    groups = list(g.groupby("group_id").indices.items())
    rng.shuffle(groups)
    groups.sort(key=lambda kv: -len(kv[1]))
    assign = {}
    filled = {p: defaultdict(float) for p in parts}
    for gid, idx in groups:
        strata = Counter(zip(g["domain"].iloc[idx], g["final_label"].iloc[idx]))
        best_p, best_score = None, None
        for p in parts:
            # deficit-weighted score: prefer the part that needs these strata most
            score = sum((quota[p].get(s, 0) - filled[p][s]) * c for s, c in strata.items())
            size_pen = (sum(filled[p].values()) / max(1, sizes[p]))
            score -= size_pen * len(idx) * 0.5
            if best_score is None or score > best_score:
                best_p, best_score = p, score
        assign[gid] = best_p
        for s, c in strata.items():
            filled[best_p][s] += c
    return g["group_id"].map(assign)


def build_gold(cfg) -> Path:
    out = prep_dir(cfg)
    g = load_gold_parts(cfg)
    ag = agreement_metrics(g)
    write_json(out / "gold_agreement.json", ag)

    sizes = {"gold_train": cfg.splits.gold_train, "gold_dev": cfg.splits.gold_dev, "gold_test": cfg.splits.gold_test}

    # --- honor fixed splits (e.g. the released gold_test), keep groups whole --
    prio = {"gold_test": 3, "gold_dev": 2, "gold_train": 1}
    fixed = g["fixed_split"] if "fixed_split" in g.columns else pd.Series([None] * len(g), index=g.index)
    fp = fixed.map(prio)
    grp_max = g.assign(_fp=fp).groupby("group_id")["_fp"].transform("max")
    inv = {v: k for k, v in prio.items()}
    g["split"] = grp_max.map(inv)  # NaN when group has no fixed member
    promoted = int((g["split"].notna() & fixed.isna()).sum())
    if promoted:
        log.info("Group consistency: %d unfixed rows follow their group's fixed split", promoted)
    n_fixed = g["split"].value_counts(dropna=True).to_dict()
    if n_fixed:
        log.info("Fixed split rows honored: %s", n_fixed)

    # A partition that already has fixed rows is externally defined - never
    # top it up with rows of other provenance (keeps gold_test pure part01).
    fixed_parts = set(fixed.dropna().unique())
    sizes_left = {k: (0 if k in fixed_parts else max(0, v - int((g["split"] == k).sum())))
                  for k, v in sizes.items()}
    rest_mask = g["split"].isna()
    rest = g[rest_mask]
    if len(rest):
        need = {k: v for k, v in sizes_left.items() if v > 0}
        if not need:
            log.warning("All split quotas already filled by fixed rows; %d leftover rows -> gold_train", len(rest))
            g.loc[rest_mask, "split"] = "gold_train"
        else:
            total_need = sum(need.values())
            if len(rest) < total_need:
                frac = len(rest) / total_need
                need = {k: max(1, int(round(v * frac))) for k, v in need.items()}
                log.warning("Only %d distributable gold rows for %d requested; scaling quotas to %s",
                            len(rest), total_need, need)
            g.loc[rest_mask, "split"] = grouped_stratified_split(rest, need, cfg.splits.seed)
    g.to_parquet(out / "gold.parquet", index=False)

    summary = {
        "counts_by_split": g["split"].value_counts().to_dict(),
        "counts_by_split_domain_label": {
            f"{s}|{d}|{l}": int(c)
            for (s, d, l), c in g.groupby(["split", "domain", "final_label"]).size().items()
        },
        "n_groups": int(g["group_id"].nunique()),
        "agreement": ag,
    }
    write_json(out / "gold_summary.json", summary)
    log.info("Gold built: %s", g["split"].value_counts().to_dict())
    return out / "gold.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    cfg = load_config()
    setup_logging(Path(cfg.paths.runs_dir) / "logs" / "gold.log")
    build_gold(cfg)


if __name__ == "__main__":
    main()
