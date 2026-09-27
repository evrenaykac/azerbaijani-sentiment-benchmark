# AzSentBench: Azerbaijani Sentiment Benchmark

AzSentBench is a leakage-audited, multi-domain sentiment benchmark for
Azerbaijani, released together with the full evaluation protocol, the audit
tooling, and the recorded results of more than six hundred training runs —
including a negative result that overturned our own earlier claim.

This repository accompanies the article *Domain-Robust Sentiment Analysis for
Azerbaijani: A Multi-Domain Benchmark and Controlled Comparison* (Aykaç, Ozkan
and Samet, IEEE Access, under review). The corpus extends the collection
introduced with our published article *Morpheme-Aware Hybrid Sentiment
Analysis for Azerbaijani* (IEEE Access, vol. 14, 2026,
doi:10.1109/ACCESS.2026.3659042); the label layers, splits and audit released
here are those of the present article. Every experimental result the article reports from its own runs can be
re-derived from what is here without re-training (the Stage-1 figures of
Section VII are quoted from the original submission); corpus-construction counts and per-batch annotation
agreement are documented rather than re-derivable, because the raw text and
the batch membership are not redistributed.

```bash
git clone https://github.com/evrenaykac/azerbaijani-sentiment-benchmark.git
cd azerbaijani-sentiment-benchmark
pip install -r requirements.txt

python scripts/verify_invariants.py      # re-derive the four leakage invariants
python scripts/verify_paper_numbers.py   # recompute the article's headline numbers
```

Both scripts read only the released files and exit non-zero if anything fails to
match. Their expected output is shown in [`docs/reproducibility.md`](docs/reproducibility.md).

---

## Why this benchmark exists

Azerbaijani sentiment analysis had data but no multi-domain evaluation whose
split could be checked for leakage. Working on our own earlier system, we found that its training-data
strategy — not the method — was producing the improvement we had reported
(+8.9 macro-F1 on the leave-one-domain-out average in the original submission).
A controlled re-execution under the released code reproduces the effect (+9.6
on Public Services against +8.8 originally, though only +2.5 on the LODO
average), and it disappears as soon as the teacher-labeled bulk comments are left out of
training:

| full adaptation recipe vs. plain XLM-R fine-tuning | earlier split, teacher-labeled bulk | earlier split, human labels only | audited split, human labels only |
|---|---|---|---|
| LODO average | **+2.5** | +0.5 | **−0.5** |
| Public Services target | **+9.6** | −1.5 | **−1.6** |
| baseline macro-F1 on that target | 0.416 | 0.734 | 0.754 |

On the same split, training on human labels only removes the gain and restores
the collapsed baseline; moving to the audited split then changes nothing of consequence.
Section VII of the article traces the mechanism. The warning sign was the
baseline itself — a competent multilingual encoder at 0.416 macro-F1 on a
three-class task — and the competing label-prior explanation can be ruled out
in seconds from stored predictions (`tools/prior_sweep.py`).

## What the benchmark contains

**Corpus.** 271,278 Azerbaijani user comments across five domains (Tech,
Finance, Social, Retail, Public Services), collected from publicly posted
comment sections.

**Gold partition.** 10,000 comments split 5,000 / 2,000 / 3,000 into
train / dev / test. The 3,000-item test partition is *fully* triple-annotated
and adjudicated: three annotators labeled every item from a written
guideline — blind for 1,743 items, with the earlier single-annotator label
visible for the other 1,257 — and a fourth team member reviewed every item,
not only the disagreements, and issued the final label. Fleiss' κ = 0.756 on the blind
10,000-item batch.

**Bulk layer.** 261,278 further comments: 87,069 labeled in a wide
single-annotator pass and 174,209 unlabeled. The labeled bulk comments make up
about 94% of the supervised training data (82,589 of 87,589 comments when all
domains are pooled); bulk text, with labels discarded, is also used for
domain-adaptive pretraining.

**Splits.** Source-grouped: for the 71.0% of comments with a source-page
identifier, whole source groups are assigned to one partition as units, so
those comments cannot straddle a split; the remaining 29.0% are singletons. Label
prior and domain mix are matched across train, dev and test to within 0.04
percentage points.

**Four leakage invariants**, all zero and all re-derivable from this repository
by code independent of the code that produced the split:

1. no duplicate normalized text anywhere in the corpus
2. no identified source group spanning two partitions
3. no evaluation text in any training partition
4. no identified evaluation source group in any training partition

## What is in this repository

```
data/
  corpus_index.csv.gz    271,278 rows: ids, domain, label, split, source group,
                         and a hash of the deduplication key (see "Text" below)
  annotations.csv        per-annotator records for the 11,250 triple-annotated
                         comments, including the adjudicator's decision
  splits/                the exact partition membership, one uid per line
  lexicon/               placeholder — see data/lexicon/README.md
results/
  runs_master.csv        one row per training run (616 runs, 20 systems), with
                         macro-F1, per-class F1, the training-pool composition
                         (gold / human-labeled bulk / silver = 0) and every
                         embedding probe
  tables/                Tables 6 and 7 of the article and supporting tables, as CSV
  stats/                 bootstrap intervals and adjusted p-values
  transfer_report.json   cross-lingual transfer arm
  llm/                   QLoRA runs (metrics, per-item test predictions, training
                         log) and, under llm/openai/, every prompted-model
                         response plus per-item predictions
  llm_openai_metrics.json  prompted GPT-4o / GPT-4o-mini, per domain and class
  lexicon_coverage.json  how much of the corpus the polarity lexicon reaches,
                         exact and prefix-backoff, per domain
  earlier_split/         the Section VII evidence: the run ledger of the
                         earlier (unaudited) evaluation, bootstrap statistics
                         for the full recipe vs. XLM-R with teacher-generated
                         and with human-only bulk labels, the earlier split's
                         label composition, and the prior-reweighting test
  ingest_report.json     the audit as it was recorded at corpus build time
src/azsent/              the experimental pipeline
scripts/                 verification and regeneration tools
docs/                    reproducibility guide, protocol, dataset card
```

### Text

The corpus contains only publicly posted comments and no author-level
attributes; source pages appear solely as salted hashes. Platform terms do not
permit us to redistribute raw comment text, so the index ships everything except
the text itself, plus `text_key_sha256` — the first 16 hex characters of the
SHA-256 digest of the pipeline's own deduplication key.

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
| **Re-train** | run the full campaign from scratch | more than 60 GPU-hours (RTX 4090 class) |

[`docs/reproducibility.md`](docs/reproducibility.md) gives the commands for each,
including which artifacts you need beyond this repository (the stored
per-item predictions, about 1.5 GB, are attached to the repository's release
rather than committed; see that document).

## Principal results

Pooled over all domains, a 278M-parameter fine-tuned encoder (mDeBERTa-v3,
five-seed mean 0.760) is close to single runs of QLoRA-adapted Qwen2.5-7B and
Qwen3-8B (0.761 and 0.759); the single Llama-3.1-8B run is about 1.5 points
higher (0.774). Under domain shift the encoder is above the adapted Qwen2.5-7B
on both held-out targets tested. Prompted GPT-4o reaches 0.677 — below every
fine-tuned system under the same pooled protocol, and below the in-domain
average of a classical tf-idf baseline (0.685, a different protocol).
Zero-shot transfer from Turkish reaches 0.379 against 0.746 for in-language
training, though a single run with Turkish intermediate training lies 1.8
points above the five-seed mean of the same recipe without it.

The negative result: on the audited split, none of the domain-adaptation
techniques evaluated — domain-adaptive pretraining, FiLM-based polarity
injection, supervised contrastive regularization, domain-adversarial training,
or their combinations — outperforms plain XLM-R fine-tuning with a
domain-specific 95% interval excluding zero. Every comparison whose interval
against plain XLM-R fine-tuning excludes zero on any target involves a change
of backbone (mDeBERTa-v3 above it on the leave-one-domain-out Tech target;
mBERT below it on two targets in each regime, as is the full recipe on an
mBERT backbone on in-domain Finance), and none of these comparisons survives
Holm adjustment.

## Citation

See [`CITATION.cff`](CITATION.cff). The article is under review at IEEE Access.

## License

Code (`src/`, `scripts/`) is released under the **MIT License**
([`LICENSE`](LICENSE)). Data and result files (`data/`, `results/`) are released
under **CC BY 4.0** ([`LICENSE-DATA`](LICENSE-DATA)).

## Contact

Yusuf Evren Aykaç — Department of Computer Engineering, Ankara Yıldırım Beyazıt
University.
Issues and questions are welcome through the GitHub issue tracker.
