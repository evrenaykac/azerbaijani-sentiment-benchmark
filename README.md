# AzSentBench: A Multi-Domain Azerbaijani Sentiment Benchmark

AzSentBench is a leakage-audited, multi-domain sentiment benchmark for
Azerbaijani, released together with the full evaluation protocol, the audit
tooling, and the recorded results of more than six hundred training runs —
including a negative result on domain-adaptation techniques and the diagnosis
of why they had appeared to help.

This repository accompanies the article *Azerbaijani Sentiment Analysis Under Domain Shift: A Multi-Domain
Benchmark and Controlled Comparison* (Aykaç, Ozkan
and Samet, IEEE Access, under review). The corpus extends the collection
introduced with our published article *Morpheme-Aware Hybrid Sentiment
Analysis for Azerbaijani* (IEEE Access, vol. 14, 2026,
doi:10.1109/ACCESS.2026.3659042); the label layers, splits and audit released
here are those of the present article. Every experimental result the article reports from its own runs can be
re-derived from what is here without re-training (the Stage-1 figures of
Section VII come from the preliminary evaluation that preceded the corpus
audit and are documented rather than re-derivable; the bootstrap intervals
and p-values of its Stages 2 and 2b are recorded in `results/earlier_split/`);
corpus-construction counts are documented rather than re-derivable, because the
raw text is not redistributed; per-batch and per-cohort annotation agreement can be
recomputed from `data/annotations.csv`, which carries the annotation batch.

Release v1.1.0, which corresponds to the article, is archived together with
the stored predictions of all reported runs at https://doi.org/10.5281/zenodo.24275188.

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
split could be checked for leakage. In a preliminary evaluation of our
adaptation recipe, conducted before the corpus audit, the recipe appeared to
improve on plain XLM-R fine-tuning by +8.9 macro-F1 on the leave-one-domain-out
average. That improvement came from the training-data strategy, not from the
method. A controlled re-execution under the released code reproduces the effect
(+9.6 on Public Services against +8.8 in the preliminary evaluation, though
only +2.5 on the LODO average), and it disappears as soon as the
teacher-labeled bulk comments are left out of training:

| full adaptation recipe vs. plain XLM-R fine-tuning | pre-audit split, teacher-labeled bulk | pre-audit split, human labels only | audited split, human labels only |
|---|---|---|---|
| LODO average | **+2.5** | +0.5 | **−0.5** |
| Public Services target | **+9.6** | −1.5 | **−1.6** |
| baseline macro-F1 on that target | 0.416 | 0.734 | 0.754 |

On the same split, training on human labels only removes the gain and restores
the collapsed baseline; moving to the audited split then changes nothing of
consequence under human labels. Two further controls on the audited split
(`results/label_controls/`, Table 15 of the article) separate the components of
the data strategy: relabeling the same bulk comments with the teacher, which
changes about a third of their labels, lowers plain fine-tuning and the recipe
alike and leaves their difference where it was, and adding the unlabeled
comments with teacher labels, the Stage-2 strategy, reproduces the direction
of the effect at about a quarter of its size, without statistical support and
without a collapsed baseline. In the conditions tested, the full-size gain therefore appeared only with
teacher-labeled comments on the pre-audit split. Section VII of the article
traces the mechanism. The warning sign was the
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
guideline — blind for 1,743 items, with the existing single-annotator label
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
                         and the annotation batch (blind or verification)
  splits/                the exact partition membership, one uid per line
  lexicon/               not vendored (published separately) — see data/lexicon/README.md
results/
  runs_master.csv        one row per training run (616 runs, 20 systems), with
                         macro-F1, per-class F1, the training-pool composition
                         (gold / human-labeled bulk / silver = 0) and every
                         embedding probe
  tables/                Tables 8 and 9 of the article and supporting tables, as CSV
  stats/                 bootstrap intervals and adjusted p-values under the
                         pre-specified seed-pair design; stats/seed_mean/ holds
                         the same 190 comparisons under a two-level bootstrap of
                         the seed-mean difference (scripts/bootstrap_designs.py)
  transfer_report.json   cross-lingual transfer arm
  llm/                   QLoRA runs (metrics, per-item test predictions, training
                         log) and, under llm/openai/, every prompted-model
                         response plus per-item predictions
  llm_openai_metrics.json  prompted GPT-4o / GPT-4o-mini, per domain and class
  lexicon_coverage.json  how much of the corpus the polarity lexicon reaches,
                         exact and prefix-backoff, per domain
  earlier_split/         the Section VII evidence: the run ledger of the
                         pre-audit evaluation, bootstrap statistics for the
                         full recipe vs. XLM-R with teacher-generated and with
                         human-only bulk labels, the pre-audit split's label
                         composition, and the prior-reweighting test
  label_controls/        the Table 15 controls on the audited split: the same
                         bulk comments relabeled by the teacher, and the Stage-2
                         strategy re-run on the audited split (36 runs, their
                         ledger, both bootstrap designs; scripts/label_controls.py)
    ingest_report.json     the audit as it was recorded at corpus build time
  near_duplicates/       character 5-gram similarity between every development
                         or test comment and the training comments: per-comment
                         best matches and counts at four thresholds
                         (tools/near_duplicates.py)
  analysis/              analyses computed from the stored predictions and the
                         annotation records: agreement with bootstrap intervals,
                         results on the blind cohort of the test partition and
                         without near-duplicate comments, two-sided p-values
                         for all 190 comparisons, the Stage-2 baseline evidence
                         and the Turkish binary reading (scripts/analysis/)
src/azsent/              the experimental pipeline
scripts/                 verification and regeneration tools
docs/                    reproducibility guide, protocol, dataset card, annotation
                         guideline, access to the comment text
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

The text is also available on request for non-commercial research, under a
data-use agreement that carries the same restrictions (no redistribution, no
attempt to identify the writers). With it, new systems can be trained and
evaluated on the fixed split; see [`docs/data_access.md`](docs/data_access.md).

## Reproducing the article

Three levels, in increasing order of cost:

| level | what it does | cost |
|---|---|---|
| **Verify** | re-derive invariants and recompute every headline number from the released files | seconds, no GPU |
| **Re-analyze** | recompute bootstrap intervals, calibration, probes and the Section VII reweighting from stored predictions | about two hours on a CPU, no GPU |
| **Re-train** | run the full campaign from scratch | more than 60 GPU-hours (RTX 4090 class) |

[`docs/reproducibility.md`](docs/reproducibility.md) gives the commands for each,
including which artifacts you need beyond this repository (the stored
per-item predictions, two archives of about 1.5 GB each, are attached to the
repository's release rather than committed; see that document).

## Principal results

Pooled over all domains, the single QLoRA-adapted Llama-3.1-8B run has the
highest score (0.774 macro-F1), fine-tuned encoders reach 0.746 to 0.760
(five-seed means of XLM-R and mDeBERTa-v3), the single Qwen2.5-7B and Qwen3-8B
runs score 0.761 and 0.759, and prompted GPT-4o reaches 0.677 with the one
prompt tested. The adapted language models are single runs of one
configuration, so the two families are at one level and are not ranked.
Zero-shot transfer from the available Turkish corpora reaches 0.379 against
0.746 for in-language training.

Plain fine-tuning of a strong multilingual encoder is the configuration the
evidence supports. Across 190 per-domain comparisons, no adaptation technique
(domain-adaptive pretraining, FiLM-based polarity injection, supervised
contrastive regularization, domain-adversarial training, or their
combinations) improves on plain XLM-R fine-tuning after Holm adjustment,
whereas multilingual BERT falls four to six points below it. The same picture
holds on the blind cohort of the test partition (1,743 comments annotated
without a visible label, all with a source identifier) and with the 159 test
comments that have a near-duplicate in the labeled training pool removed
(`results/analysis/`).

## Citation

See [`CITATION.cff`](CITATION.cff). The archived release v1.1.0 has the DOI
[10.5281/zenodo.24275188](https://doi.org/10.5281/zenodo.24275188). The article is under review at IEEE Access.

## License

Code (`src/`, `scripts/`) is released under the **MIT License**
([`LICENSE`](LICENSE)). Data and result files (`data/`, `results/`) are released
under **CC BY 4.0** ([`LICENSE-DATA`](LICENSE-DATA)).

## Contact

Yusuf Evren Aykaç — Department of Computer Engineering, Ankara Yıldırım Beyazıt
University.
Issues and questions are welcome through the GitHub issue tracker.
