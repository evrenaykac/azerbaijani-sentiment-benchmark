# Reproducibility guide

Three levels of reproduction are supported. The first two need no GPU and no
model downloads; the third re-runs the full campaign.

---

## Level 1 — Verify (seconds, no GPU)

Everything at this level runs against the files in this repository alone.

```bash
pip install -r requirements.txt
python scripts/verify_invariants.py
python scripts/verify_paper_numbers.py
```

### Expected output — `verify_invariants.py`

```
index: 271,278 rows from data/corpus_index.csv.gz
  gold partition: 10,000   training pool: 253,140   evaluation: 5,000
[PASS] 1. duplicate normalized text ....... 0
[PASS] 2. source groups spanning splits ... 0
[PASS] 3. evaluation text in training ..... 0
[PASS] 4. evaluation group in training .... 0
[info ] groups spanning gold/bulk *training* layers: 26 (by design; both layers are training material)

source-group identifier coverage: 71.0%  (paper reports 71.0%)

all four invariants hold
```

Two notes on that output, both deliberate:

**The 26 groups.** A source group may hold rows in both the gold and the bulk
*training* layers, because the gold partition is drawn from the same pool. Both
layers are training material, so this is by design and is not leakage. The
invariant that matters — no group spanning train, dev or test — is zero. We
print the figure rather than suppress it.

**71.0% coverage.** The source-group guarantee is conditional on identifier
coverage. Where an identifier exists, whole groups move as units; the remaining
comments are singletons, and for those no grouping stronger than the exact and
near-duplicate text keys can be enforced. Same-source comments without
identifiers could in principle fall on opposite sides of a split. This is stated
in Section III-E of the article and repeated here so that the guarantee is not
read as stronger than it is.

### Expected output — `verify_paper_numbers.py`

29 checks, all passing: the main-table averages for both regimes, the confidence
interval of the study's only significant positive comparison, the Holm-adjusted
family p-values, a joint Holm adjustment across all 190 per-domain comparisons,
the four prompted-LLM scores, the three cross-lingual transfer scores, the
corpus accounting, and the annotation counts. The script prints
`29/29 checks passed` and exits 0.

The joint-Holm line deserves a word. The article summarizes per-domain p-values
by their median before Holm adjustment, which is a conservative reporting device
rather than the basis of any positive claim. Applying Holm–Bonferroni jointly
across all 190 per-domain one-sided comparisons instead leaves every adjusted
p-value at 1.0, so no conclusion depends on how the family is formed. That
alternative is computed by the script, not asserted.

---

## Level 2 — Re-analyze (minutes, no GPU)

This level recomputes derived quantities from the stored per-item predictions:
bootstrap intervals, calibration, embedding probes, and the prior-reweighting
forensics of Section VII.

The predictions are approximately 1.5 GB and are therefore **not** in this Git
repository. Obtain `predictions.tar.gz` from the archived release (see the
repository's Releases page, or the DOI recorded in `CITATION.cff` once
assigned), then:

```bash
tar xzf predictions.tar.gz -C runs/
python -m azsent.postproc                     # rebuilds results/tables from stored runs
python -m azsent.run_stats                    # bootstrap intervals + Holm adjustment
python tools/prior_sweep.py --runs runs/ --out results/tables/prior_sweep_lodo.csv \
    --regime lodo --systems full xlmr_ft --baseline xlmr_ft
```

Paths are read from `configs/default.yaml`; point its `paths.runs_dir` at the
directory you extracted into.

`prior_sweep.py` is the Section VII analysis. It reweights stored predictions to
a target label prior and recomputes macro-F1 exactly, with no Monte-Carlo noise
and no re-training — which is why the diagnosis of a mis-specified split takes
seconds rather than a GPU day. Its output reproduces Table 10: the gap does not
shrink under prior matching, it grows, which is what eliminated the label-prior
hypothesis.

---

## Level 3 — Re-train (≈ 40 GPU-hours)

Full campaign. You will need the comment text (Level 3 only), a GPU, and model
downloads.

### 1. Restore the text

```bash
python scripts/regenerate_text.py --source your_comments.csv --text-column text
```

See the README on why the text is not distributed and how the join works. The
script reports match coverage explicitly; a partial corpus will not reproduce
the published numbers exactly.

### 2. Provide the lexicon

The polarity-injection systems need the SentiAzNet polarity lexicon; see
[`../data/lexicon/README.md`](../data/lexicon/README.md).

### 3. Preflight

```bash
python -m azsent.preflight            # add --skip-net to check cached assets only
```

The preflight runs 33 checks and refuses to build training pools if any leakage
invariant fails. It also runs an annotation-integrity check that fails if
inter-annotator agreement is implausibly high (κ > 0.95), if annotator
disagreement is under 2%, or if any annotator matches the final label on more
than 99.5% of items. That check exists because an earlier annotation export we
received had been generated rather than collected, and these thresholds are what
caught it.

### 4. Run

```bash
python -m azsent.runner --blocks core --list      # show the job plan first
python -m azsent.runner --blocks core             # ~600 runs: prep, teachers, DAPT,
                                                  # both regimes, postproc, stats, report
python -m azsent.runner --blocks llm,transfer     # adapted and prompted LLMs, transfer
```

`--blocks all` runs everything including the ablation and sensitivity grids.
`--blocks core_fast` answers the main claim in a few hours without the full
tables. On a multi-GPU machine, `--shard I N --gpu K` splits a block across
parallel workers. Systems and seed budgets are declared in
`configs/systems.yaml`; paths and hyperparameters in `configs/default.yaml`.

Seeds are declared per system in the configuration rather than chosen after
seeing results: five seeds (13, 21, 42, 87, 100) for every system a claim rests
on, three (13, 42, 100) for context rows. The tf-idf baseline is run under the
same five seeds; its fit is deterministic and all five coincide exactly, which
is why the tables show no deviation for it.

### Hardware and cost

The published campaign ran on 3× RTX 4090 (24 GB each) and took roughly 40
GPU-hours in total, including the QLoRA adaptations of three 7–8B models. The
encoder runs alone fit comfortably on a single 24 GB card.

The prompted-LLM arm requires an OpenAI API key and costs a few dollars at the
pinned snapshots (`gpt-4o-2024-08-06`, `gpt-4o-mini-2024-07-18`). Decoding is
deterministic (temperature 0) and every request and response is logged. Outputs
that fail to parse are counted as errors rather than silently dropped; in the
published run, none occurred.

---

## Numerical safeguards

Each of these corresponds to a failure we hit during development, and each is in
the pipeline because of it:

- **fp32 forcing for fp16 checkpoints.** mDeBERTa-v3 ships fp16 weights that
  produce non-finite losses under mixed precision on this task; the pipeline
  forces fp32 for affected checkpoints.
- **Abort on non-finite loss.** A run that goes non-finite fails loudly instead
  of writing a plausible-looking result.
- **Feature-logic-versioned token cache.** The cache key includes the tokenizer,
  the sequence length *and* a version of the feature-extraction logic, so
  changing feature code cannot silently reuse stale cached features.
- **Architectural self-test.** `smoke/selftest_arch.py` asserts that FiLM
  conditioning is the identity at initialization and that mid-encoder and
  post-encoder injection produce different activations — the check that would
  have caught the placement discrepancy in the earlier version of this work.

## Environment

Developed on Python 3.11 with PyTorch 2.x and CUDA 12.x. `requirements.txt`
covers Levels 1 and 2; `requirements-llm.txt` and `requirements-transfer.txt`
add what the LLM and transfer arms need. Exact versions used for the published
campaign are pinned in those files.
