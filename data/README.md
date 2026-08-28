# Data

## Files

### `corpus_index.csv.gz` — 271,278 rows

One row per comment. Everything the experiments need except the text itself.

| column | description |
|---|---|
| `uid` | stable comment identifier |
| `domain` | `Tech`, `Finance`, `Social`, `Retail`, `Public` |
| `label` | `negative` / `neutral` / `positive`; empty for the 174,209 unlabelled bulk comments |
| `split` | `train`, `dev`, `test` |
| `gold_role` | `gold_train`, `gold_dev`, `gold_test`; empty for bulk-layer rows |
| `video_id_hash` | salted hash of the source page; empty for singletons (29.0% of rows) |
| `source` | which raw export the row came from |
| `source_row` | row index within that export |
| `n_annotators` | 3 for triple-annotated comments, 1 for the wide pass, 0 for unlabelled |
| `text_key_sha256` | first 16 hex characters of SHA-256 over the normalized deduplication key |

### `annotations.csv` — 11,250 rows

Per-annotator records for every triple-annotated comment.

| column | description |
|---|---|
| `uid` | joins to the index |
| `domain` | domain of the comment |
| `ann1`, `ann2`, `ann3` | the three independent annotators' labels |
| `adjudicator` | the fourth team member's decision |
| `final_label` | the label used in all experiments (equal to `adjudicator`) |
| `gold_role` | which gold partition the comment belongs to |

The adjudicator reviewed **every** item, not only the disagreements, and
departed from the annotator majority on 1.02% of the blind batch. Agreement
statistics exclude the adjudicator: their decision is the final label by
construction, and including it would inflate every coefficient.

Two batches with different protocols are pooled in this file but reported
separately in the article, because they are not comparable:

- **Blind batch** (10,000 items): three annotators, no prior label visible.
  Fleiss' κ = 0.756; pairwise Cohen's κ from 0.750 to 0.762; unanimity 76.9%.
- **Verification batch** (1,257 items): completed triple annotation of the test
  partition with the existing single-annotator label visible on the sheet. This
  makes it adjudicated verification rather than blind re-annotation. Its κ of
  0.752 is consistent with the blind batch, and its higher unanimity (80.5%) is
  what a visible prior label would predict.

Seven of the 10,000 blind-batch items were later removed by corpus-level
deduplication, which is why 11,250 comments are retained rather than 11,257.

### `splits/`

Partition membership as plain uid lists, one per line: `gold_train.txt`,
`gold_dev.txt`, `gold_test.txt`, `bulk_train.txt`, `bulk_dev.txt`.

Leave-one-domain-out folds are derived from these by holding out one domain at a
time; the pipeline builds them and re-verifies every invariant before training.

### `lexicon/`

Placeholder. See [`lexicon/README.md`](lexicon/README.md).

## Why there is no text here

The corpus contains only publicly posted comments, and no author-level
attributes were collected — source pages appear solely as salted hashes. Raw
redistribution of user text nonetheless conflicts with platform terms, so the
text is not in this repository.

`text_key_sha256` is what keeps the release verifiable rather than merely
descriptive. It is the SHA-256 digest of the pipeline's own deduplication key:
NFKC normalization, Azerbaijani-aware lowercasing (`İ`→`i`, `I`→`ı`, which
`str.lower()` gets wrong), URL and mention stripping, punctuation and emoji
removed, whitespace squeezed. Two comments share a hash exactly when the
pipeline considers them duplicates, so the duplicate-text and text-overlap
invariants are checkable without the text — and anyone holding the same public
comments can rejoin them:

```bash
python scripts/regenerate_text.py --source your_comments.csv --text-column text
```

The join is on normalized text, not on our identifiers, so a third party who
re-collects the comments independently can rebuild the corpus without trusting
our id scheme.

## Composition

| domain | comments | share |
|---|---|---|
| Social | 84,206 | 31.04% |
| Tech | 57,784 | 21.30% |
| Finance | 47,473 | 17.50% |
| Public | 47,249 | 17.42% |
| Retail | 34,566 | 12.74% |
| **total** | **271,278** | **100.00%** |

Domain shares range from 31.0% to 12.7%: the corpus is multi-domain, not
domain-balanced. What *is* matched is composition across partitions — the label
prior and the domain mix agree between train, dev and test to within 0.04
percentage points, which is the property the evaluation depends on. Exact
per-domain counts are in `results/split_counts.json` and are recomputed by
`scripts/verify_paper_numbers.py`.

## Known limitations

- **Source-group coverage is 71.0%.** Where an identifier exists, whole groups
  move as units. The remaining comments are singletons; for those, no grouping
  stronger than the text keys can be enforced, so same-source comments without
  identifiers could in principle fall on opposite sides of a split.
- **89.2% of gold-train carries a single annotator's label.** Only 10.8% of the
  training partition is triple-annotated. The *test* partition is fully
  triple-annotated and adjudicated, which is where it matters for evaluation,
  but training-label noise is real.
- **Two annotation protocols.** See the batch note above; they are reported
  separately and never pooled as though comparable.
- **Neutral is the hardest class** for every system in every regime, and
  per-domain agreement is lowest in Finance (κ = 0.685), where neutral–negative
  boundaries blur.

## Licence

CC BY 4.0 — see [`../LICENSE-DATA`](../LICENSE-DATA). That licence covers the
compiled dataset released here; it grants no rights in the underlying comment
text, which is not distributed and remains subject to platform terms.
