"""SentiAzNet lexicon adapter.

Reads whatever file the user drops into <data_dir>/lexicon/ (csv/tsv/xlsx/json)
and produces a normalized {word -> polarity in [-1,1]} mapping. Column roles are
auto-detected: a word/term column plus either a numeric polarity column or a
categorical label column (neg/neu/pos).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .normalize import az_lower, normalize_text
from .utils import log

_WORD_CANDS = ["word", "term", "az_term", "token", "soz", "söz", "kelime", "lemma", "entry", "text"]
_SCORE_CANDS = ["polarity", "score", "sentiment_score", "polarity_score", "value", "weight", "skor", "sentiment_skoru"]
_LABEL_CANDS = ["label", "sentiment", "polarity_label", "sentiment_etiketi", "class", "etiket"]

_LABEL2SCORE = {
    "negative": -1.0, "neg": -1.0, "-1": -1.0,
    "neutral": 0.0, "neu": 0.0, "notr": 0.0, "0": 0.0,
    "positive": 1.0, "pos": 1.0, "1": 1.0,
}


def _read_any(path: Path) -> pd.DataFrame:
    sfx = path.suffix.lower()
    if sfx == ".xlsx":
        return pd.read_excel(path)
    if sfx == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return pd.DataFrame({"word": list(obj.keys()), "polarity": list(obj.values())})
        return pd.DataFrame(obj)
    sep = "\t" if sfx in (".tsv", ".txt") else ","
    return pd.read_csv(path, sep=sep)


def load_lexicon(data_dir: str | Path, subdir: str = "lexicon") -> dict[str, float]:
    lx_dir = Path(data_dir) / subdir
    files = sorted([p for p in lx_dir.glob("*") if p.suffix.lower() in (".csv", ".tsv", ".txt", ".xlsx", ".json")]) if lx_dir.exists() else []
    if not files:
        raise FileNotFoundError(
            f"No lexicon file found in {lx_dir}. Put the SentiAzNet lexicon "
            f"(word + polarity columns; csv/xlsx/json) into this folder."
        )
    merged: dict[str, float] = {}
    for f in files:
        df = _read_any(f)
        low = {c.lower().strip(): c for c in df.columns}
        wcol = next((low[c] for c in _WORD_CANDS if c in low), None)
        scol = next((low[c] for c in _SCORE_CANDS if c in low), None)
        lcol = next((low[c] for c in _LABEL_CANDS if c in low), None)
        if wcol is None:
            # single-column file: words only -> +1? refuse, ambiguous
            raise ValueError(f"{f.name}: cannot find a word column (columns: {list(df.columns)})")
        if "positive" in low and "negative" in low:
            # SentiWordNet-style probability triple -> scalar polarity in [-1,1]
            scores = pd.to_numeric(df[low["positive"]], errors="coerce") - pd.to_numeric(df[low["negative"]], errors="coerce")
        elif scol is not None:
            scores = pd.to_numeric(df[scol], errors="coerce")
            mx = np.nanmax(np.abs(scores.to_numpy(dtype=float))) or 1.0
            if mx > 1.0:  # rescale e.g. [-5,5] lexicons into [-1,1]
                scores = scores / mx
        elif lcol is not None:
            scores = df[lcol].map(lambda v: _LABEL2SCORE.get(str(v).strip().lower(), np.nan))
        else:
            raise ValueError(f"{f.name}: no polarity score or label column found (columns: {list(df.columns)})")
        n_before = len(merged)
        for w, s in zip(df[wcol], scores):
            if pd.isna(w) or pd.isna(s):
                continue
            key = az_lower(str(w).strip())
            if key:
                merged[key] = float(np.clip(s, -1.0, 1.0))
        log.info("Lexicon %s: +%d entries (total %d)", f.name, len(merged) - n_before, len(merged))
    return merged


def coverage_report(lex: dict[str, float], texts: list[str], domains: list[str]) -> dict:
    """Token/comment hit rates by domain (paper Table 4)."""
    from collections import defaultdict
    from .normalize import words

    tok_hit = defaultdict(lambda: [0, 0])
    com_hit = defaultdict(lambda: [0, 0])
    pol_counts = defaultdict(lambda: [0, 0, 0])  # neg neu pos hits
    for t, d in zip(texts, domains):
        ws = words(normalize_text(t))
        hits = 0
        for w in ws:
            tok_hit[d][1] += 1
            s = lex.get(w.strip(".,!?…\"'()[]"))
            if s is not None:
                tok_hit[d][0] += 1
                hits += 1
                if s < -0.33:
                    pol_counts[d][0] += 1
                elif s > 0.33:
                    pol_counts[d][2] += 1
                else:
                    pol_counts[d][1] += 1
        com_hit[d][1] += 1
        com_hit[d][0] += int(hits > 0)
    out = {}
    for d in tok_hit:
        th = tok_hit[d][0] / max(1, tok_hit[d][1])
        ch = com_hit[d][0] / max(1, com_hit[d][1])
        tot = max(1, sum(pol_counts[d]))
        out[d] = {
            "token_hit_rate": round(th, 4),
            "comment_hit_rate": round(ch, 4),
            "neg_hit_share": round(pol_counts[d][0] / tot, 4),
            "neu_hit_share": round(pol_counts[d][1] / tot, 4),
            "pos_hit_share": round(pol_counts[d][2] / tot, 4),
        }
    return out
