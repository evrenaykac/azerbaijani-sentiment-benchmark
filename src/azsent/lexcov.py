"""Report how much of the corpus the sentiment lexicon actually reaches.

In the first full run the polarity-conditioning component contributed almost
nothing, and the reason turned out to be coverage rather than mechanism: exact
surface matching fired on 3.6% of word tokens and left 43.8% of comments with no
lexical signal at all, so the FiLM layer was conditioning on a feature that was
zero nearly everywhere. Coverage is therefore measured up front and reported
alongside the ablation, so the numbers can be read correctly.
"""
from __future__ import annotations

import argparse

import pandas as pd

from .config import load_config, prep_dir
from .lexicon import load_lexicon
from .utils import log, setup_logging, write_json

STRIP = ".,!?;:\"()[]{}'«»…-"


def coverage(texts, uni: set[str], min_prefix: int) -> dict:
    tot = exact = prefix = with_hit = 0
    for t in texts:
        hits = 0
        for w0 in str(t).split():
            w = w0.strip(STRIP)
            if not w:
                continue
            tot += 1
            if w in uni:
                exact += 1
                hits += 1
            else:
                for L in range(len(w) - 1, min_prefix - 1, -1):
                    if w[:L] in uni:
                        prefix += 1
                        hits += 1
                        break
        if hits:
            with_hit += 1
    n = max(1, tot)
    return {
        "tokens": tot,
        "exact_hits": exact, "exact_pct": round(100 * exact / n, 2),
        "prefix_extra_hits": prefix, "prefix_extra_pct": round(100 * prefix / n, 2),
        "combined_pct": round(100 * (exact + prefix) / n, 2),
        "comments_with_any_hit_pct": round(100 * with_hit / max(1, len(texts)), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20000)
    args = ap.parse_args()
    cfg = load_config()
    out = prep_dir(cfg)
    setup_logging(out / "lexcov.log")
    lex = load_lexicon(cfg.paths.data_dir, cfg.paths.lexicon_subdir)
    uni = {k for k in lex if " " not in k}
    gold = pd.read_parquet(out / "gold.parquet")
    bulk = pd.read_parquet(out / "bulk.parquet")
    minp = int(cfg.get("lexicon", {}).get("min_prefix_len", 4))
    res = {
        "lexicon_unigrams": len(uni),
        "lexicon_bigrams": len(lex) - len(uni),
        "match_mode": cfg.get("lexicon", {}).get("match", "prefix"),
        "min_prefix_len": minp,
        "gold_overall": coverage(gold["text"].tolist()[: args.sample], uni, minp),
        "gold_by_domain": {
            d: coverage(g["text"].tolist()[: args.sample // 5], uni, minp)
            for d, g in gold.groupby("domain")
        },
        # the unlabeled bulk layer is a different population (longer, noisier),
        # so the paper should quote whichever one it actually describes
        "bulk_overall": coverage(
            bulk["text"].sample(n=min(args.sample, len(bulk)), random_state=42).tolist(), uni, minp),
    }
    write_json(out / "lexicon_coverage.json", res)
    o = res["gold_overall"]
    b = res["bulk_overall"]
    log.info("Sozluk kapsamasi (gold): tam %.2f%% | prefix ile %.2f%% | en az bir eslesen yorum %.1f%%",
             o["exact_pct"], o["combined_pct"], o["comments_with_any_hit_pct"])
    log.info("Sozluk kapsamasi (bulk): tam %.2f%% | prefix ile %.2f%% | en az bir eslesen yorum %.1f%%",
             b["exact_pct"], b["combined_pct"], b["comments_with_any_hit_pct"])


if __name__ == "__main__":
    main()
