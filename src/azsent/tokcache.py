"""Persistent tokenization + feature cache.

Every run in the queue tokenizes the same comments and recomputes the same
token-level polarity/morphology features, once per epoch, on the main thread.
This module does that work ONCE per (tokenizer, max_len) and stores it as flat
numpy arrays that every later run memory-maps.

Stored per comment:
  ids   : subword ids (already truncated to max_len)
  wid   : subword -> word index (-1 for special tokens)
  feats : word-level feature matrix (n_words x 8, float16)

The full 8-dimensional feature block is always cached; feature subsets
(pol / morph / all) are applied at collate time by masking, so one cache
serves every system variant.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import prep_dir
from .features import D_FEAT, FeatureExtractor
from .utils import log


def tokenizer_fingerprint(tokenizer) -> str:
    """Identity of the actual tokenizer, so a different vocabulary can never
    silently reuse another tokenizer's cache."""
    probe = tokenizer("salam d\u00fcnya \u0259la deyil", add_special_tokens=True)["input_ids"]
    raw = f"{len(tokenizer)}|{getattr(tokenizer, 'name_or_path', '')}|{probe}"
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


def cache_key(tokenizer_key: str, max_len: int, lexicon_fingerprint: str, tok_fp: str = "",
              feat_ver: str = "") -> str:
    # feat_ver guards the failure we hit once: changing how features are computed
    # without changing the lexicon left the key identical, so the run silently
    # reused the old features and the experiment measured nothing.
    h = hashlib.sha1(
        f"{tokenizer_key}|{max_len}|{lexicon_fingerprint}|{tok_fp}|{feat_ver}".encode()
    ).hexdigest()[:10]
    return f"{tokenizer_key}_len{max_len}_{h}"


def lexicon_fingerprint(lex: dict) -> str:
    if not lex:
        return "nolex"
    h = hashlib.sha1()
    for k in sorted(lex):
        h.update(k.encode("utf-8"))
        h.update(f"{lex[k]:.4f}".encode())
    return h.hexdigest()[:10]


class TokenCache:
    """Flat-array cache with uid -> row lookup."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.ids_flat = z["ids_flat"]
        self.ids_off = z["ids_off"]
        self.wid_flat = z["wid_flat"]
        self.feat_flat = z["feat_flat"]
        self.feat_off = z["feat_off"]
        self.lengths = z["lengths"]
        uids = z["uids"]
        self.row = {u: i for i, u in enumerate(uids.tolist())}

    def __contains__(self, uid: str) -> bool:
        return uid in self.row

    def get(self, uid: str):
        i = self.row[uid]
        a, b = self.ids_off[i], self.ids_off[i + 1]
        c, d = self.feat_off[i], self.feat_off[i + 1]
        return self.ids_flat[a:b], self.wid_flat[a:b], self.feat_flat[c:d]

    def length(self, uid: str) -> int:
        return int(self.lengths[self.row[uid]])


def build_cache(cfg, tokenizer, tokenizer_key: str, lex: dict, max_len: int,
                force: bool = False) -> Path:
    """Tokenize + featurize every corpus comment once; write the cache file."""
    fp = lexicon_fingerprint(lex)
    tfp = tokenizer_fingerprint(tokenizer)
    out = Path(cfg.paths.runs_dir) / "cache" / (cache_key(tokenizer_key, max_len, fp, tfp,
                                 feat_ver=FeatureExtractor.FEATURE_LOGIC_VERSION
                                 + "|" + str(cfg.get("lexicon", {}).get("match", "prefix"))
                                 + str(cfg.get("lexicon", {}).get("min_prefix_len", 4))) + ".npz")
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    p = prep_dir(cfg)
    bulk = pd.read_parquet(p / "bulk.parquet")[["uid", "text"]]
    gold = pd.read_parquet(p / "gold.parquet")[["uid", "text"]]
    df = pd.concat([bulk, gold], ignore_index=True).drop_duplicates(subset=["uid"]).reset_index(drop=True)
    log.info("Building token cache for %s (%d comments, max_len=%d) ...", tokenizer_key, len(df), max_len)

    fx = FeatureExtractor(lex, lang="az", subset="all",
                          match=str(cfg.get("lexicon", {}).get("match", "prefix")),
                          min_prefix_len=int(cfg.get("lexicon", {}).get("min_prefix_len", 4)))
    n = len(df)
    ids_parts, wid_parts, feat_parts = [], [], []
    ids_off = np.zeros(n + 1, dtype=np.int64)
    feat_off = np.zeros(n + 1, dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int32)

    CH = 2000
    texts = df["text"].tolist()
    for s in range(0, n, CH):
        chunk = texts[s : s + CH]
        wordlists, featmats = [], []
        for t in chunk:
            ws, F = fx(t)
            wordlists.append(ws)
            featmats.append(F.astype(np.float32))
        enc = tokenizer(wordlists, is_split_into_words=True, truncation=True,
                        max_length=max_len, padding=False)
        for j in range(len(chunk)):
            ids = np.asarray(enc["input_ids"][j], dtype=np.int32)
            wid = np.asarray([-1 if w is None else w for w in enc.word_ids(j)], dtype=np.int16)
            i = s + j
            ids_parts.append(ids)
            wid_parts.append(wid)
            feat_parts.append(featmats[j])
            ids_off[i + 1] = ids_off[i] + len(ids)
            feat_off[i + 1] = feat_off[i] + featmats[j].shape[0]
            lengths[i] = len(ids)
        if (s + CH) % 50000 < CH:
            log.info("  cache %d/%d", min(s + CH, n), n)

    np.savez(
        out,
        uids=np.array(df["uid"].tolist()),
        ids_flat=np.concatenate(ids_parts).astype(np.int32),
        ids_off=ids_off,
        wid_flat=np.concatenate(wid_parts).astype(np.int16),
        feat_flat=np.concatenate(feat_parts).astype(np.float32),
        feat_off=feat_off,
        lengths=lengths,
    )
    mb = out.stat().st_size / 2**20
    log.info("Token cache written: %s (%.0f MB, mean length %.1f subwords)", out, mb, lengths.mean())
    return out
