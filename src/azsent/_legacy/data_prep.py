"""Step 1: Excel -> parquet with normalization, deduplication and group ids.

Reviewer coverage:
  * R1-3: exact + near-duplicate detection, fully reported (dedup_report.json)
  * R1-2: source group ids (video_id where available) for grouped splitting
  * Cross-domain duplicate texts are collapsed to a single representative so
    the same surface form can never sit on both sides of a domain split.

Outputs (in <runs>/prepared/):
  corpus.parquet     one row per retained comment
  removed.parquet    every removed row with a removal reason
  dedup_report.json  counts by reason/domain, procedure parameters
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, prep_dir
from .normalize import char_shingles, dedup_key, normalize_text, words
from .utils import log, setup_logging, stable_uid, write_json, sha256_file

DOM_CANON = {
    "Technology & Digital Services": "Tech",
    "Finance & Business": "Finance",
    "Social Life & Entertainment": "Social",
    "Retail & Lifestyle": "Retail",
    "Public Services": "Public",
}


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _near_dup_clusters(keys: list[str], num_perm: int, threshold: float, k: int) -> np.ndarray:
    """MinHash-LSH near-duplicate clustering over dedup keys.

    Returns an int array: cluster id per row (cluster id = representative row index).
    """
    from datasketch import MinHash, MinHashLSH

    n = len(keys)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    uf = _UnionFind(n)
    mhs: list[MinHash] = []
    for i, key in enumerate(keys):
        m = MinHash(num_perm=num_perm)
        for sh in char_shingles(key, k):
            m.update(sh.encode("utf-8"))
        for j in lsh.query(m):
            uf.union(int(j), i)
        lsh.insert(str(i), m)
        mhs.append(m)
        if (i + 1) % 20000 == 0:
            log.info("  minhash %d/%d", i + 1, n)
    return np.array([uf.find(i) for i in range(n)], dtype=np.int64)


def prepare(cfg, excel_path: str | None = None, sample: int | None = None) -> Path:
    out = prep_dir(cfg)
    excel = Path(excel_path) if excel_path else Path(cfg.paths.data_dir) / cfg.paths.excel_name
    log.info("Reading %s", excel)
    df = pd.read_excel(excel)
    src_sha = sha256_file(excel)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)

    # --- canonicalize ---------------------------------------------------------
    df["domain"] = df["domain"].map(DOM_CANON).fillna(df["domain"])
    unknown = sorted(set(df["domain"]) - set(DOM_CANON.values()))
    if unknown:
        raise ValueError(f"Unknown domains in Excel: {unknown}")
    df["text_raw"] = df["yorum"].astype(str)
    df["text"] = df["text_raw"].map(normalize_text)
    df["key"] = df["text_raw"].map(dedup_key)
    df["n_words"] = df["text"].map(lambda t: len(words(t)))
    df["label"] = df["sentiment_etiketi"].where(df["etiket_durumu"] == "etiketli")
    df["has_human_label"] = df["label"].notna()
    df["video_id"] = df["video_id"].astype("string")
    df["uid"] = [stable_uid(k, str(s)) for k, s in zip(df["key"], df["sira_no"])]

    removed_frames = []

    def _remove(mask: pd.Series, reason: str) -> None:
        nonlocal df
        if mask.any():
            part = df.loc[mask].copy()
            part["removal_reason"] = reason
            removed_frames.append(part)
            df = df.loc[~mask].copy()

    n0 = len(df)
    _remove(df["key"].str.len() < cfg.dedup.min_chars, "too_short_after_normalization")

    # --- exact duplicates -----------------------------------------------------
    # Keep one representative per identical key. Priority: has video_id > has
    # human label > longer raw text; ties broken by sira_no for determinism.
    df["_prio"] = (
        df["video_id"].notna().astype(int) * 4
        + df["has_human_label"].astype(int) * 2
        + (df["text_raw"].str.len() / 1e9)
    )
    df = df.sort_values(["key", "_prio", "sira_no"], ascending=[True, False, True])
    exact_dup_mask = df.duplicated(subset=["key"], keep="first")
    n_exact_groups = int(df.duplicated(subset=["key"], keep=False).sum() - exact_dup_mask.sum())
    cross_dom_keys = df[df.duplicated(subset=["key"], keep=False)].groupby("key")["domain"].nunique()
    n_cross_domain_exact = int((cross_dom_keys > 1).sum())
    _remove(exact_dup_mask, "exact_duplicate")
    df = df.reset_index(drop=True)

    # --- near duplicates ------------------------------------------------------
    n_near_removed = 0
    n_cross_domain_near = 0
    if cfg.dedup.near_dup:
        log.info("Near-duplicate MinHash-LSH over %d unique texts ...", len(df))
        clusters = _near_dup_clusters(
            df["key"].tolist(), cfg.dedup.minhash_num_perm, cfg.dedup.minhash_threshold, cfg.dedup.shingle_size
        )
        df["_cluster"] = clusters
        sizes = df.groupby("_cluster")["uid"].transform("size")
        multi = df[sizes > 1]
        n_cross_domain_near = int((multi.groupby("_cluster")["domain"].nunique() > 1).sum())
        df = df.sort_values(["_cluster", "_prio", "sira_no"], ascending=[True, False, True])
        near_mask = df.duplicated(subset=["_cluster"], keep="first")
        n_near_removed = int(near_mask.sum())
        _remove(near_mask, "near_duplicate")
        df = df.drop(columns=["_cluster"]).reset_index(drop=True)

    df = df.drop(columns=["_prio"])

    # --- group ids for grouped splitting -------------------------------------
    # video_id where present; otherwise the comment is its own singleton group.
    df["group_id"] = np.where(df["video_id"].notna(), "vid_" + df["video_id"].fillna(""), "solo_" + df["uid"])

    keep_cols = [
        "uid", "sira_no", "kaynak", "kaynak_detay", "domain", "video_id", "group_id",
        "text_raw", "text", "key", "n_words", "label", "has_human_label",
    ]
    df = df[keep_cols].reset_index(drop=True)
    df.to_parquet(out / "corpus.parquet", index=False)
    removed = pd.concat(removed_frames, ignore_index=True) if removed_frames else pd.DataFrame()
    if len(removed):
        removed.to_parquet(out / "removed.parquet", index=False)

    grp_cov = df.groupby("domain")["video_id"].apply(lambda s: float(s.notna().mean())).to_dict()
    report = {
        "source_excel": str(excel),
        "source_sha256": src_sha,
        "rows_in": n0,
        "rows_out": len(df),
        "removed_total": int(len(removed)),
        "removed_by_reason": removed["removal_reason"].value_counts().to_dict() if len(removed) else {},
        "exact_dup_groups": n_exact_groups,
        "exact_dup_groups_cross_domain": n_cross_domain_exact,
        "near_dup_removed": n_near_removed,
        "near_dup_clusters_cross_domain": n_cross_domain_near,
        "procedure": {
            "normalization": "NFKC, az-aware lowercase, URL/mention removal, repeated-punct squeeze, whitespace collapse",
            "exact": "SHA-1 over punctuation-stripped normalized text",
            "near": f"MinHash-LSH, char {cfg.dedup.shingle_size}-gram shingles, {cfg.dedup.minhash_num_perm} permutations, Jaccard>={cfg.dedup.minhash_threshold}",
            "representative_priority": "has video_id > has human label > longer text",
        },
        "labels_kept": df["label"].value_counts(dropna=False).to_dict(),
        "human_labeled": int(df["has_human_label"].sum()),
        "unlabeled": int((~df["has_human_label"]).sum()),
        "video_id_coverage_by_domain": grp_cov,
        "rows_by_domain": df["domain"].value_counts().to_dict(),
    }
    write_json(out / "dedup_report.json", report)
    log.info("Prepared corpus: %d rows kept, %d removed. Report: %s", len(df), len(removed), out / "dedup_report.json")
    return out / "corpus.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", default=None)
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    setup_logging(Path(cfg.paths.runs_dir) / "logs" / "data_prep.log")
    prepare(cfg, args.excel, args.sample)


if __name__ == "__main__":
    main()
