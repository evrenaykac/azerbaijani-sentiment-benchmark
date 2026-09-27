"""Azerbaijani-aware text normalization and tokenization.

Normalization is used for (a) dedup keys, (b) lexicon lookup and cue matching.
Model inputs use a lighter normalization that preserves case-insensitive
matching behaviour identical to the feature extractor (words are lowercased
consistently for both).
"""
from __future__ import annotations

import re
import unicodedata

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_WS_RE = re.compile(r"\s+")
_REPEAT_PUNCT_RE = re.compile(r"([!?.,])\1{2,}")

# Azerbaijani/Turkish lowercase: İ -> i, I -> ı  (standard lower() breaks these)
_AZ_LOWER_MAP = str.maketrans({"İ": "i", "I": "ı"})


def az_lower(text: str) -> str:
    return text.translate(_AZ_LOWER_MAP).lower()


def normalize_text(text: str, for_dedup: bool = False) -> str:
    """NFKC + az-lower + url/mention scrub + whitespace/punct squeeze."""
    t = unicodedata.normalize("NFKC", str(text))
    t = _URL_RE.sub(" ", t)
    t = _MENTION_RE.sub(" ", t)
    t = _REPEAT_PUNCT_RE.sub(r"\1\1", t)
    t = az_lower(t)
    if for_dedup:
        # strip everything except letters/digits for a robust dedup key
        t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = _WS_RE.sub(" ", t).strip()
    return t


def dedup_key(text: str) -> str:
    return normalize_text(text, for_dedup=True)


_TOKEN_RE = re.compile(r"\S+")


def words(text: str) -> list[str]:
    """Whitespace word split of the (already normalized) text."""
    return _TOKEN_RE.findall(text)


def char_shingles(text: str, k: int = 5) -> set[str]:
    t = text.replace(" ", "_")
    if len(t) <= k:
        return {t}
    return {t[i : i + k] for i in range(len(t) - k + 1)}
