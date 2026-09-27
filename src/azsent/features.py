"""Token-level polarity + surface-morphology features (paper Table 5, d_f = 8).

Feature order (fixed):
  0 lex_polarity   scalar in [-1,1]
  1 lex_hit        binary
  2 neg_suffix     binary   -ma/-mə, -mır/-mir/-mur/-mür, -madı/-mədi, -mayıb/-məyib ...
  3 neg_explicit   binary   deyil, yox, heç
  4 intensifier    binary   çox, lap, tam, həddindən artıq, olduqca ...
  5 diminisher     binary   bir az, azca, o qədər də ...
  6 emoji_pos      binary
  7 emoji_neg      binary

Feature subsets used by ablations:
  pol   -> dims {0,1}
  morph -> dims {2,3,4,5,6,7}
  all   -> all 8

Languages: 'az' (default, full inventory), 'kk', 'uz' (heuristic cue sets used
only by the Turkic-transfer recipe experiments; lexicon dims stay zero there
unless a lexicon for that language is supplied).
"""
from __future__ import annotations

import re

import numpy as np

from .normalize import normalize_text, words

D_FEAT = 8
FEAT_NAMES = ["lex_polarity", "lex_hit", "neg_suffix", "neg_explicit",
              "intensifier", "diminisher", "emoji_pos", "emoji_neg"]
SUBSETS = {
    "all": list(range(8)),
    "pol": [0, 1],
    "morph": [2, 3, 4, 5, 6, 7],
}

_POS_EMOJI = set("😀😃😄😁😆😊🙂😍🥰😘😗☺️🤗🤩😻👍👏💪❤️🧡💛💚💙💜🖤🤍🤎💖💕💞💓💗✨🎉🥳😂🤣")
_NEG_EMOJI = set("😠😡🤬😞😔😟😕🙁☹️😣😖😫😩🥺😢😭😤😨😰😥💔👎🤢🤮😒🙄😑")
_POS_EMOTICON = re.compile(r"^[:;=]-?[)\]dDpP]+$|^\^\^$|^<3$")
_NEG_EMOTICON = re.compile(r"^[:;=]'?-?[(\[/\\]+$|^</3$")

_CUES = {
    "az": {
        # negation suffix patterns on the *end* of a word (verb negation, informal variants included)
        "neg_suffix": re.compile(
            r"(ma|mə|mır|mir|mur|mür|mıram|mirəm|muram|mürəm|madı|mədi|mayıb|məyib|mayacaq|məyəcək|masın|məsin|mazdı|məzdi|maz|məz|mayın|məyin|ması|məsi)$"
        ),
        "neg_word": {"deyil", "deyildi", "deyiləm", "yox", "yoxdur", "yoxdu", "heç"},
        "neg_suffix_exceptions": {"amma", "firma", "forma", "reklama", "proqrama", "sistemə", "cəmi"},
        "intens": {"çox", "lap", "tam", "olduqca", "həddindən", "tamamilə", "son", "əla", "müthiş", "dəhşət"},
        "intens_bigrams": {("həddindən", "artıq"), ("çox", "çox")},
        "dimin": {"azca", "birtəhər", "nisbətən"},
        "dimin_bigrams": {("bir", "az"), ("bir", "qədər"), ("o", "qədər"), ("az", "da")},
    },
    "kk": {  # Kazakh (Cyrillic) - heuristic, used only in the transfer recipe
        "neg_suffix": re.compile(r"(ма|ме|ба|бе|па|пе|майды|мейді|маған|меген|мады|меді)$"),
        "neg_word": {"емес", "жоқ", "ешқашан"},
        "intens": {"өте", "тым", "аса", "керемет"},
        "intens_bigrams": set(),
        "dimin": {"аздап", "сәл", "біраз"},
        "dimin_bigrams": set(),
    },
    "uz": {  # Uzbek (Latin) - heuristic
        "neg_suffix": re.compile(r"(ma|mas|madi|magan|maydi|masin)$"),
        "neg_word": {"emas", "yo'q", "yoq", "hech"},
        "intens": {"juda", "eng", "zo'r", "nihoyatda", "g'oyat"},
        "intens_bigrams": set(),
        "dimin": {"ozgina", "biroz", "sal"},
        "dimin_bigrams": set(),
    },
}

_STRIP = ".,!?…\"'()[]«»:;"


class FeatureExtractor:
    FEATURE_LOGIC_VERSION = "2"   # bump whenever word_features changes; the token
                                  # cache keys on it, otherwise a logic change is
                                  # silently served stale features.

    def __init__(self, lexicon: dict[str, float] | None, lang: str = "az", subset: str = "all",
                 match: str = "prefix", min_prefix_len: int = 4):
        # split lexicon into unigram and bigram entries (longer n-grams are dropped)
        self.lex: dict[str, float] = {}
        self.bi: dict[str, float] = {}
        n_long = 0
        for k, v in (lexicon or {}).items():
            spaces = k.count(" ")
            if spaces == 0:
                self.lex[k] = v
            elif spaces == 1:
                self.bi[k] = v
            else:
                n_long += 1
        self.n_dropped_long = n_long
        self.cues = _CUES[lang]
        self.match = match
        self.min_prefix_len = int(min_prefix_len)
        self.subset = subset
        mask = np.zeros(D_FEAT, dtype=np.float32)
        mask[SUBSETS[subset]] = 1.0
        self.mask = mask

    def word_features(self, ws: list[str]) -> np.ndarray:
        n = len(ws)
        F = np.zeros((n, D_FEAT), dtype=np.float32)
        for i, w0 in enumerate(ws):
            # emoji/emoticon checks use the raw token (emoticons are pure punctuation)
            if any(ch in _POS_EMOJI for ch in w0) or _POS_EMOTICON.match(w0):
                F[i, 6] = 1.0
            if any(ch in _NEG_EMOJI for ch in w0) or _NEG_EMOTICON.match(w0):
                F[i, 7] = 1.0
            w = w0.strip(_STRIP)
            if not w:
                continue
            s = self.lex.get(w)
            if s is None and self.match == "prefix" and len(w) > self.min_prefix_len:
                # Azerbaijani is agglutinative: "yaxsi" is in the lexicon but
                # "yaxsidir" is not, so exact matching reaches only a fraction of
                # the tokens a polarity feature should fire on.
                for L in range(len(w) - 1, self.min_prefix_len - 1, -1):
                    s = self.lex.get(w[:L])
                    if s is not None:
                        break
            if s is not None:
                F[i, 0] = s
                F[i, 1] = 1.0
            if (len(w) > 3 and w not in self.cues.get("neg_suffix_exceptions", ())
                    and self.cues["neg_suffix"].search(w)):
                F[i, 2] = 1.0
            if w in self.cues["neg_word"]:
                F[i, 3] = 1.0
            if w in self.cues["intens"]:
                F[i, 4] = 1.0
            if w in self.cues["dimin"]:
                F[i, 5] = 1.0
            if i + 1 < n:
                nxt = ws[i + 1].strip(_STRIP)
                if (w, nxt) in self.cues["intens_bigrams"]:
                    F[i, 4] = F[i + 1 if i + 1 < n else i, 4] = 1.0
                if (w, nxt) in self.cues["dimin_bigrams"]:
                    F[i, 5] = 1.0
        # bigram lexicon entries: assign polarity to both members (override unigram)
        if self.bi:
            for i in range(n - 1):
                key = ws[i].strip(_STRIP) + " " + ws[i + 1].strip(_STRIP)
                s = self.bi.get(key)
                if s is not None:
                    F[i, 0] = F[i + 1, 0] = s
                    F[i, 1] = F[i + 1, 1] = 1.0
        return F * self.mask

    def __call__(self, text: str) -> tuple[list[str], np.ndarray]:
        ws = words(normalize_text(text))
        if not ws:
            ws = ["."]
        return ws, self.word_features(ws)
