#!/usr/bin/env python3
"""Near-duplicate check of the evaluation comments against the training comments.

The released leakage audit covers exact duplicates after normalization (the
deduplication key) and identified source pages. This script adds a character
n-gram check: for every gold-dev and gold-test comment it finds the most similar
training comment by Jaccard similarity of character 5-gram shingle sets, computed
on the same normalized key the pipeline uses for deduplication.

It needs the comment text, so it runs on a machine that holds it. The outputs
contain identifiers and similarity values only -- no text.

Usage (from the benchmark repository root, or pass --repo):

    python near_duplicates.py --text path/to/comments.parquet --out near_dup_out
    python near_duplicates.py --text a.xlsx b.csv --text-column text --out near_dup_out
    python near_duplicates.py --text comments.parquet --check     # coverage only

Text files may be csv, xlsx or parquet. If a file has a ``uid`` column that
matches the index it is used directly; otherwise each row is matched through the
hash of its deduplication key (``text_key_sha256``), exactly as
``scripts/regenerate_text.py`` does.

Outputs in --out:
    near_dup_eval.csv       one row per evaluation comment (identifiers, similarities)
    near_dup_summary.json   counts by threshold, partition, domain and source coverage
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

K = 5                                  # shingle length (characters), as in azsent.normalize.char_shingles
THRESHOLDS = (0.95, 0.90, 0.80, 0.70)
MIN_SHINGLES = 8                       # shorter keys are reported separately: a handful of shingles makes Jaccard unstable

# --- the pipeline's normalization, reproduced so that the script stands alone ---------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_WS_RE = re.compile(r"\s+")
_REPEAT_PUNCT_RE = re.compile(r"([!?.,])\1{2,}")
_AZ_LOWER_MAP = str.maketrans({"İ": "i", "I": "ı"})


def dedup_key(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text))
    t = _URL_RE.sub(" ", t)
    t = _MENTION_RE.sub(" ", t)
    t = _REPEAT_PUNCT_RE.sub(r"\1\1", t)
    t = t.translate(_AZ_LOWER_MAP).lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    return _WS_RE.sub(" ", t).strip()


def text_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def shingles(key: str, k: int = K) -> set[str]:
    t = key.replace(" ", "_")
    if len(t) <= k:
        return {t}
    return {t[i:i + k] for i in range(len(t) - k + 1)}


# --- input ------------------------------------------------------------------------------------
def read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def load_text(paths: list[Path], text_col: str, uid_col: str, idx: pd.DataFrame) -> pd.Series:
    """Return a Series uid -> deduplication key for every index row that could be matched."""
    by_uid: dict[str, str] = {}
    hash_to_uid = dict(zip(idx["text_key_sha256"], idx["uid"]))
    known = set(idx["uid"])
    for p in paths:
        if not p.exists():
            sys.exit(f"text file not found: {p}")
        df = read_any(p)
        col = text_col if text_col in df.columns else next(
            (c for c in ("text", "comment", "yorum", "content") if c in df.columns), None)
        if col is None:
            sys.exit(f"{p.name}: no text column (tried {text_col!r}, text, comment, yorum, content); "
                     f"columns are {list(df.columns)[:15]}")
        df = df[df[col].notna()]
        n_uid = n_hash = 0
        use_uid = uid_col in df.columns and df[uid_col].astype(str).isin(known).mean() > 0.5
        for row_uid, text in zip(df[uid_col].astype(str) if use_uid else [None] * len(df), df[col].astype(str)):
            key = dedup_key(text)
            if use_uid and row_uid in known:
                by_uid.setdefault(row_uid, key); n_uid += 1
            else:
                u = hash_to_uid.get(text_hash(key))
                if u is not None:
                    by_uid.setdefault(u, key); n_hash += 1
        print(f"  {p.name}: {len(df):,} rows, column {col!r}; matched {n_uid:,} by uid and {n_hash:,} by text hash")
    return pd.Series(by_uid, name="key")


# --- similarity -------------------------------------------------------------------------------
def build(keys: list[str], vocab: dict[str, int] | None):
    """Sparse binary shingle matrix restricted to `vocab` (built from `keys` when None) and shingle counts."""
    grow = vocab is None
    if grow:
        vocab = {}
    indptr, indices, sizes = [0], [], np.empty(len(keys), dtype=np.int32)
    for n, key in enumerate(keys):
        sh = shingles(key)
        sizes[n] = len(sh)
        if grow:
            for s in sh:
                j = vocab.get(s)
                if j is None:
                    j = vocab[s] = len(vocab)
                indices.append(j)
        else:
            indices.extend(vocab[s] for s in sh if s in vocab)
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.int32)
    return (sparse.csr_matrix((data, np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)),
                              shape=(len(keys), len(vocab))), sizes, vocab)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=Path("."), help="benchmark repository root (holds data/corpus_index.csv.gz)")
    ap.add_argument("--text", type=Path, nargs="+", required=True, help="file(s) holding the comment text")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--uid-column", default="uid")
    ap.add_argument("--out", type=Path, default=Path("near_dup_out"))
    ap.add_argument("--chunk", type=int, default=200, help="evaluation comments per block")
    ap.add_argument("--check", action="store_true", help="report text coverage and stop")
    a = ap.parse_args()

    index_path = a.repo / "data" / "corpus_index.csv.gz"
    if not index_path.exists():
        sys.exit(f"index not found: {index_path} (run from the repository root or pass --repo)")
    idx = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    print(f"index rows: {len(idx):,}")
    keys = load_text(a.text, a.text_column, a.uid_column, idx)
    idx["key"] = idx["uid"].map(keys)
    idx["has_text"] = idx["key"].notna()

    is_eval = idx["gold_role"].isin(["gold_dev", "gold_test"])
    is_train = idx["split"] == "train"
    cov = {"index_rows": int(len(idx)), "with_text": int(idx["has_text"].sum()),
           "eval_rows": int(is_eval.sum()), "eval_with_text": int((is_eval & idx["has_text"]).sum()),
           "train_rows": int(is_train.sum()), "train_with_text": int((is_train & idx["has_text"]).sum()),
           "train_labeled_rows": int((is_train & (idx["label"] != "")).sum()),
           "train_labeled_with_text": int((is_train & (idx["label"] != "") & idx["has_text"]).sum())}
    print("coverage:", json.dumps(cov))
    if cov["eval_with_text"] < cov["eval_rows"] or cov["train_with_text"] < 0.99 * cov["train_rows"]:
        print("WARNING: text is missing for part of the corpus; near-duplicates of the missing rows cannot be found.")
    if a.check:
        return 0
    if cov["eval_with_text"] == 0 or cov["train_with_text"] == 0:
        sys.exit("no text matched; check --text-column / --uid-column")

    ev = idx[is_eval & idx["has_text"]].reset_index(drop=True)
    tr = idx[is_train & idx["has_text"]].reset_index(drop=True)
    t0 = time.time()
    Xe, ne, vocab = build(ev["key"].tolist(), None)
    Xt, nt, _ = build(tr["key"].tolist(), vocab)
    XtT = Xt.T.tocsr()
    print(f"shingles: {len(vocab):,} distinct in the evaluation comments; matrices built in {time.time() - t0:.0f}s")

    tr_dom = tr["domain"].to_numpy(); tr_lab = tr["label"].to_numpy(); tr_uid = tr["uid"].to_numpy()
    tr_labeled = tr_lab != ""
    tr_keys = tr["key"].tolist(); ev_keys = ev["key"].tolist()
    rows = []
    for s in range(0, len(ev), a.chunk):
        P = (Xe[s:s + a.chunk] @ XtT).tocsr()
        for r in range(P.shape[0]):
            i = s + r
            lo, hi = P.indptr[r], P.indptr[r + 1]
            cols, inter = P.indices[lo:hi], P.data[lo:hi].astype(np.float64)
            jac = inter / (ne[i] + nt[cols] - inter)
            same = tr_dom[cols] == ev.at[i, "domain"]
            lab = tr_labeled[cols]
            rec = {"uid": ev.at[i, "uid"], "gold_role": ev.at[i, "gold_role"], "domain": ev.at[i, "domain"],
                   "has_source_id": int(ev.at[i, "video_id_hash"] != ""), "n_shingles": int(ne[i]),
                   "n_words": len(ev_keys[i].split())}
            for name, mask in (("labeled_same_domain", lab & same), ("labeled_other_domain", lab & ~same),
                               ("unlabeled_same_domain", ~lab & same), ("unlabeled_other_domain", ~lab & ~same)):
                if mask.any():
                    k = int(np.argmax(np.where(mask, jac, -1.0)))
                    c = int(cols[k])
                    # exact Jaccard on the full shingle sets (the matrix is exact already; this is a guard)
                    A, B = shingles(ev_keys[i]), shingles(tr_keys[c])
                    j = len(A & B) / len(A | B)
                    rec[f"j_{name}"] = round(j, 4); rec[f"uid_{name}"] = tr_uid[c]
                    if name.startswith("labeled"):
                        rec[f"same_label_{name}"] = int(tr_lab[c] == ev.at[i, "label"])
                else:
                    rec[f"j_{name}"] = 0.0; rec[f"uid_{name}"] = ""
                    if name.startswith("labeled"):
                        rec[f"same_label_{name}"] = ""
            for t in THRESHOLDS:
                rec[f"n_labeled_ge{int(t * 100)}"] = int((lab & (jac >= t)).sum())
                rec[f"n_any_ge{int(t * 100)}"] = int((jac >= t).sum())
            rows.append(rec)
        print(f"  {min(s + a.chunk, len(ev)):,} / {len(ev):,} evaluation comments  ({time.time() - t0:.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    out["j_labeled_any"] = out[["j_labeled_same_domain", "j_labeled_other_domain"]].max(axis=1)
    out["j_any"] = out[[c for c in out.columns if c.startswith("j_") and c != "j_labeled_any"]].max(axis=1)
    a.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out / "near_dup_eval.csv", index=False)

    def block(df: pd.DataFrame) -> dict:
        ok = df[df["n_shingles"] >= MIN_SHINGLES]
        d = {"n": int(len(df)), "n_with_at_least_%d_shingles" % MIN_SHINGLES: int(len(ok))}
        for t in THRESHOLDS:
            tag = f"ge{int(t * 100)}"
            d[tag] = {"labeled_train_any_domain": int((ok["j_labeled_any"] >= t).sum()),
                      "labeled_train_same_domain": int((ok["j_labeled_same_domain"] >= t).sum()),
                      "labeled_train_other_domain": int((ok["j_labeled_other_domain"] >= t).sum()),
                      "any_train_comment": int((ok["j_any"] >= t).sum())}
            hit = ok[ok["j_labeled_any"] >= t]
            if len(hit):
                best_same = np.where(hit["j_labeled_same_domain"] >= hit["j_labeled_other_domain"],
                                     hit["same_label_labeled_same_domain"], hit["same_label_labeled_other_domain"])
                d[tag]["best_labeled_match_has_same_label"] = round(float(pd.to_numeric(pd.Series(best_same), errors="coerce").mean()), 4)
        return d

    summary = {"k": K, "thresholds": list(THRESHOLDS), "min_shingles": MIN_SHINGLES, "coverage": cov}
    for role in ("gold_test", "gold_dev"):
        part = out[out["gold_role"] == role]
        summary[role] = {"all": block(part),
                         "with_source_id": block(part[part["has_source_id"] == 1]),
                         "without_source_id": block(part[part["has_source_id"] == 0]),
                         "by_domain": {d: block(part[part["domain"] == d]) for d in sorted(part["domain"].unique())}}
    (a.out / "near_dup_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    t = summary["gold_test"]["all"]
    print(f"\ngold-test: {t['n']:,} comments; with a labeled training comment at Jaccard >= 0.90: "
          f"{t['ge90']['labeled_train_any_domain']:,}; >= 0.80: {t['ge80']['labeled_train_any_domain']:,}; "
          f">= 0.70: {t['ge70']['labeled_train_any_domain']:,}")
    print(f"written: {a.out / 'near_dup_eval.csv'} and {a.out / 'near_dup_summary.json'} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
