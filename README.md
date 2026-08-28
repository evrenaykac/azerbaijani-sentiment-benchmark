# Azerbaijani Sentiment Benchmark (SentiAzNet)

A leakage-audited, multi-domain sentiment benchmark for Azerbaijani, together
with the full evaluation protocol, the audit tooling, and the recorded results
of more than six hundred training runs — including a negative result that
overturned our own earlier published claim.

This repository accompanies the article *SentiAzNet: A Leakage-Audited,
Multi-Domain Azerbaijani Sentiment Benchmark and a Controlled Comparison of
Encoders, Adapted LLMs, and Cross-Lingual Transfer*. Everything reported in the
article can be re-derived from what is here, without re-training.

```bash
git clone https://github.com/<your-account>/azerbaijani-sentiment-benchmark.git
cd azerbaijani-sentiment-benchmark
pip install -r requirements.txt

python scripts/verify_invariants.py      # re-derive the four leakage invariants
python scripts/verify_paper_numbers.py   # recompute the article's headline numbers
```

Both scripts read only the released files and exit non-zero if anything fails to
match. Their expected output is shown in [`docs/reproducibility.md`](docs/reproducibility.md).

---

## Why this benchmark exists

Azerbaijani sentiment analysis had resources but no evaluation whose split could
be trusted. Working on our own earlier system, we found that the split — not the
method — was producing the improvement we had reported. Rebuilding the
evaluation reversed the conclusion:

| | earlier split | audited split |
|---|---|---|
| full adaptation recipe vs. plain fine-tuning (LODO average) | **+2.5** | **−0.5** |
| same, on the hardest target (Public Services) | **+9.6** | **−1.6** |
| baseline macro-F1 on that target | 0.416 | 0.754 |

The collapsed baseline in the left-hand column is the signature of a
mis-specified split. Section VII of the article traces the mechanism; the
analysis runs in seconds on the stored predictions, which is the point — this
class of error is cheap to detect once the predictions are kept.

## What the benchmark contains

**Corpus.** 271,278 Azerbaijani user comments across five domains (Tech,
Finance, Social, Retail, Public Services), collected from publicly posted
comment sections.

**Gold partition.** 10,000 comments split 5,000 / 2,000 / 3,000 into
train / dev / test. The 3,000-item test partition is *fully* triple-annotated
and adjudicated: three annotators labelled every item independently from a
written guideline, and a fourth team member reviewed every item — not only the
disagreements — and issued the final label. Fleiss' κ = 0.756 on the blind
10,000-item batch.

**Bulk layer.** 261,278 further comments: 87,069 labelled in a wide
single-annotator pass and 174,209 unlabelled, available for domain-adaptive
pretraining.

**Splits.** Source-grouped: whole source groups are assigned to one partition as
units, so comments from the same source page cannot straddle a split. Label
prior and domain mix are matched across train, dev and test to within 0.04
percentage points.

**Four leakage invariants**, all zero and all re-derivable from this repository
by code independent of the code that produced the split:

1. no duplicate normalized text anywhere in the corpus
2. no source group spanning two partitions
3. no evaluation text in any training partition
4. no evaluation source group in any training partition

## What is in this repository

```
data/
  corpus_index.csv.gz    271,278 rows: ids, domain, label, split, source group,
                         and a hash of the normalized text (see "Text" below)
  annotations.csv        per-annotator records for the 11,250 triple-annotated
                         comments, including the adjudicator's decision
  splits/                the exact partition membership, one uid per line
  lexicon/               placeholder — see data/lexicon/README.md
results/
  runs_master.csv        one row per training run (601 runs, 20 systems), with
                         macro-F1, per-class F1, and every embedding probe
  tables/                the article's result tables as CSV
  stats/                 bootstrap intervals and adjusted p-values
  transfer_report.json   cross-lingual transfer arm
  llm_openai_metrics.json  prompted GPT-4o / GPT-4o-mini, per domain and class
  ingest_report.json     the audit as it was recorded at corpus build time
src/azsent/              the experimental pipeline
scripts/                 verification and regeneration tools
docs/                    reproducibility guide, protocol, dataset card
```

### Text

The corpus contains only publicly posted comments and no author-level
attributes; source pages appear solely as salted hashes. Platform terms do not
permit us to redistribute raw comment text, so the index ships everything except
the text itself, plus `text_key_sha256` — the SHA-256 digest of the pipeline's
own deduplication key.

That hash is what makes the release verifiable rather than merely descriptive:
duplicate-text and text-overlap invariants can be checked without the text, and
anyone holding the same public comments can rejoin them:

```bash
python scripts/regenerate_text.py --source your_comments.csv --text-column text
```

The join is on the normalized text itself, not on an opaque identifier, so a
third party who re-collects the comments can rebuild the corpus without trusting
our id scheme.

## Reproducing the article

Three levels, in increasing order of cost:

| level | what it does | cost |
|---|---|---|
| **Verify** | re-derive invariants and recompute every headline number from the released files | seconds, no GPU |
| **Re-analyze** | recompute bootstrap intervals, calibration, probes and the Section VII reweighting from stored predictions | minutes, no GPU |
| **Re-train** | run the full campaign from scratch | ≈ 40 GPU-hours on 3× RTX 4090 |

[`docs/reproducibility.md`](docs/reproducibility.md) gives the commands for each,
including which artifacts you need beyond this repository (the stored
per-item predictions are ~1.5 GB and are archived separately; see that document).

## Principal results

A 278M-parameter fine-tuned encoder (mDeBERTa-v3) matches QLoRA-adapted 7–8B
language models when all domains are pooled (0.760 against 0.761 and 0.760) and
outperforms the adapted model tested on both held-out targets tested. Prompted
GPT-4o reaches 0.677 — below every fine-tuned system under the same pooled
protocol, and not clearly above a classical tf-idf baseline. Zero-shot transfer
from Turkish reaches 0.379 against 0.746 for in-language training, though
Turkish intermediate training adds 1.8 points.

The negative result: on the audited split, none of the domain-adaptation
techniques evaluated — domain-adaptive pretraining, FiLM-based polarity
injection, supervised contrastive regularization, domain-adversarial training,
or their combinations — significantly outperforms plain fine-tuning. Backbone
choice is the only intervention that survives significance testing.

## Citation

See [`CITATION.cff`](CITATION.cff). The article is under review at IEEE Access;
this section will be updated when it appears.

## Licence

Code (`src/`, `scripts/`) is released under the **MIT Licence**
([`LICENSE`](LICENSE)). Data and result files (`data/`, `results/`) are released
under **CC BY 4.0** ([`LICENSE-DATA`](LICENSE-DATA)).

## Contact

Yusuf Evren Aykaç — Department of Computer Engineering, Ankara University.
Issues and questions are welcome through the GitHub issue tracker.
