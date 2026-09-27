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
note: the remaining comments are singletons; for those, no grouping
      stronger than the canonical text key above can be enforced (Section III-E).

all four invariants hold
```

Two notes on that output, both deliberate:

**The 26 groups.** A source group may hold rows in both the gold and the bulk
*training* layers, because the gold partition is drawn from the same pool. Both
layers are training material, so this is by design and is not leakage. The
invariant that matters — no group spanning train, dev or test — is zero. We
print the figure rather than suppress it.

**71.0% coverage.** Where a source-page identifier exists, whole groups move
as units, and invariants 2 and 4 are checked over those identified groups. The
remaining comments are singletons: no grouping stronger than the canonical text
key (exact and near-exact variants) applies to them, and invariants 1 and 3
cover them. Section III-E of the article states the guarantee in this form.

### Expected output — `verify_paper_numbers.py`

60 checks, all passing: the main-table averages for both regimes, the confidence
interval of mDeBERTa-v3 over XLM-R on the LODO Tech target (one of the two
positive comparisons on the audited benchmark whose interval excludes zero), the Holm-adjusted
family p-values, a joint Holm adjustment across all 190 per-domain comparisons,
the pooled five-seed encoder means and deviations, the ledger size and the
absence of silver rows from every training pool, the Section VII measurements
on the earlier split (with teacher-generated and with human-only bulk labels,
including the mBERT-backbone replication) and the prior-reweighting test, the four prompted-LLM scores,
the five QLoRA scores, the three cross-lingual transfer scores, the corpus
accounting, the lexicon-coverage figures, a re-parse of every prompted-model
response, and the annotation counts. The script prints
`60/60 checks passed` and exits 0.

The joint-Holm line deserves a word. The article summarizes per-domain p-values
by their median before Holm adjustment, which is an exploratory summary
rather than the basis of any positive claim. Applying Holm–Bonferroni jointly
across all 190 per-domain one-sided comparisons instead leaves every adjusted
p-value at 1.0, so the null result for the adaptation techniques does not
depend on how the family is formed. That
alternative is computed by the script, not asserted.

---

## Level 2 — Re-analyze (minutes, no GPU)

This level recomputes derived quantities from the stored per-item predictions:
bootstrap intervals, calibration, embedding probes, and the prior-reweighting
forensics of Section VII.

The predictions are approximately 1.5 GB and are therefore **not** in this Git
repository. Obtain `predictions.tar.gz` from the repository's Releases page, then:

```bash
tar xzf predictions.tar.gz -C runs/
python -m azsent.postproc                     # rebuilds results/tables from stored runs
python -m azsent.run_stats                    # bootstrap intervals + Holm adjustment
python tools/prior_sweep.py --runs runs/ --out results/tables/prior_sweep_lodo.csv \
    --regime lodo --baseline xlmr_ft \
    --systems xlmr_ft full full_midinject xlmr_supcon xlmr_dapt mdeberta_ft mdeberta_dapt
```

Paths are read from `configs/default.yaml`; point its `paths.runs_dir` at the
directory that holds the run folders (one folder per run id).

`prior_sweep.py` is the reweighting machinery of Section VII. It reweights
stored predictions to a target label prior and recomputes macro-F1 exactly,
with no Monte-Carlo noise and no re-training — which is why the label-prior
test takes seconds rather than a GPU day.

The Section VII ladder itself rests on the earlier (unaudited) evaluation. Its
summary evidence is released in `results/earlier_split/`: the run ledger of
that campaign (`runs_master_earlier_split.csv`, 600 runs, `silver_frac` = 1.0 with
`pool_mode` gold+bulk marking runs trained with teacher-generated bulk labels and the `frac000` tag
marking the same systems trained on human labels only), the hierarchical
bootstrap of the full recipe against XLM-R fine-tuning under both conditions,
the earlier split's label composition (`gold_summary_earlier_split.json`) and
the Table 11 reweighting (`prior_reweight_test2_earlier_split.json`). The
per-item predictions of that campaign are attached to the same release as
`predictions_earlier_split.tar.gz`; with them, every Section VII number except
the Stage-1 figures quoted from the original submission is recomputable from
stored predictions. The ledger lists 28 of the 30
human-label LODO runs: `lodo.Social.xlmr_ft.s42.frac000` and
`lodo.Social.xlmr_ft.s100.frac000` are missing from it, but their predictions
are in that archive and enter the Section VII statistics, which use all nine
seed pairs on every domain.

---

## Level 3 — Re-train (more than 60 GPU-hours)

Full campaign. You will need the comment text (Level 3 only), a GPU, and model
downloads.

### 1. Restore the text

```bash
python scripts/regenerate_text.py --source your_comments.csv --text-column text
```

See the README on why the text is not distributed and how the join works. The
script reports match coverage explicitly; a partial corpus will not reproduce
the reported numbers exactly.

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
than 99.5% of items. Label columns derived from the final label, rather than collected
independently, cannot pass it.

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
seeing results: five seeds (13, 21, 42, 87, 100) for the full recipe, its
primary baseline and its contrastive ablation in both regimes, for the classical
baselines in-domain, for the mid-injection and backbone variants of the recipe
and mDeBERTa with domain-adaptive pretraining under LODO, and for the pooled encoders; three for the remaining systems (13, 21, 42
in-domain; 13, 42, 100 under LODO). The bootstrap uses whatever seed pairs exist and records their number. The tf-idf baseline is run under the
same five seeds; its fit is deterministic and all five coincide exactly, which
is why the tables show no deviation for it.

### Hardware and cost

The reported campaign ran on RTX 4090 cards (24 GB each). The recorded training
time sums to about 50 GPU-hours for the 616 runs of the ledger and about 12 for
the five QLoRA adaptations; domain-adaptive pretraining, the teachers and the
transfer arm come on top. The
encoder runs alone fit comfortably on a single 24 GB card.

The prompted-LLM arm requires an OpenAI API key and costs a few dollars at the
pinned snapshots (`gpt-4o-2024-08-06`, `gpt-4o-mini-2024-07-18`). Decoding is
deterministic (temperature 0). Every response is logged and released under
`results/llm/openai/` together with per-item predictions; the prompts themselves
contain comment text, so they are not stored verbatim, but they are fully
determined by the template in `src/azsent/llm_api.py`, the batch order (20
comments per request, test-partition order) and the fixed-seed exemplar draw.
Outputs that fail to parse are counted as errors rather than silently dropped;
in the reported run, none occurred — `verify_paper_numbers.py` re-parses all
600 responses to confirm it.

---

## Numerical safeguards

Each of these corresponds to a failure we hit during development, and each is in
the pipeline because of it:

- **fp32 forcing for fp16 checkpoints.** mDeBERTa-v3 ships fp16 weights that
  produce non-finite losses under mixed precision on this task; the pipeline
  loads every encoder in fp32, whatever precision its checkpoint ships in.
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
add what the LLM and transfer arms need, and `requirements-core.txt` covers the
encoder campaign. `transformers` is pinned to the version the campaign used
(5.15.1); the other entries are minimum versions.
