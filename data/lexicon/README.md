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

Any `.csv`, `.tsv`, `.txt`, `.xlsx` or `.json` file in this directory is read
(the directory name is `paths.lexicon_subdir` in `configs/default.yaml`), so the
file name itself does not matter.

## Expected format

One row per entry. Column roles are auto-detected by `src/azsent/lexicon.py`:
a word/term column (`word`, `term`, `token`, `lemma`, ...) plus either a numeric
polarity column (`polarity`, `score`, ...; rescaled into [-1, 1] if needed) or a
categorical label column (`label`, `sentiment`, ...; neg/neu/pos mapped to
-1/0/+1). Entries of one or two words are used; longer n-grams are dropped.

The feature extractor matches with prefix back-off (longest lexicon prefix of at
least four characters), which is what lets a single entry cover Azerbaijani's
agglutinative surface forms. Section V of the article reports lexicon coverage
on the gold partition: 4.3% of tokens match exactly, 6.0% with prefix back-off,
and 53% of comments contain at least one match; `results/lexicon_coverage.json`
carries the per-domain figures.

## A note on the name

*SentiAzNet* is the name of the polarity lexicon above. The corpus and
benchmark released in this repository are called *AzSentBench*.
