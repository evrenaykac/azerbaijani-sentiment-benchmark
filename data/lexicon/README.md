# Polarity lexicon (not included)

The polarity-injection systems read token-level polarity scores from the
SentiAzNet polarity lexicon, published separately:

> Aykaç, Y. E., Samet, R., Pashayev, A. B., and Sabziev, E. N. (2025).
> *SentiAzNet: A Polarity Lexicon for Azerbaijani.* In 2025 10th International
> Conference on Computer Science and Engineering (UBMK), IEEE.
> DOI: [10.1109/UBMK67458.2025.11207041](https://doi.org/10.1109/UBMK67458.2025.11207041)

The lexicon is a distinct artifact with its own citation, so it is not vendored
into this repository.

## What is needed

Only Level 3 of [`../../docs/reproducibility.md`](../../docs/reproducibility.md)
— re-training — requires it. Verification and re-analysis (Levels 1 and 2) run
without it, because they work from stored results.

## Where to put it

Place the lexicon file in this directory:

```
data/lexicon/SentiAzNet_2025_v1.xlsx
```

and point `paths.lexicon` in `configs/default.yaml` at it if you use a different
name or location.

## Expected format

One row per entry, with at least:

| column | description |
|---|---|
| term | the lexical entry, in its surface or stem form |
| polarity | a signed scalar; negative for negative polarity, positive for positive |

The feature extractor matches with prefix back-off, which is what lets a single
entry cover Azerbaijani's agglutinative surface forms. Section V of the article
reports lexicon coverage on this corpus: 5.7% of tokens are matched by prefix.

## A note on the name

*SentiAzNet* names two artifacts from this group: the polarity lexicon above,
and the corpus and benchmark released here. Throughout the article, unqualified
*SentiAzNet* means the corpus and benchmark; the lexicon is always called the
*SentiAzNet lexicon*.
