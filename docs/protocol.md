# Evaluation protocol

A summary of Section IV of the article, for readers who want the rules without
the prose. The pipeline enforces all of it; nothing here is advisory.

## Regimes

| regime | training pool | evaluation |
|---|---|---|
| **In-domain** | one domain | held-out test items of that domain |
| **LODO** | four domains | the fifth, held out entirely |
| **Pooled** | all five domains | the full test partition |

Under leave-one-domain-out the held-out target contributes nothing: not to
training, not to development, not to domain-adaptive pretraining, and not to any
teacher model. The pipeline re-verifies this before every fold.

## Metric

Macro-averaged F1 over the three classes, preferred to accuracy because neutral
is both the smallest class and the one systems most often collapse. Per-class F1
is reported for every system in the appendix, since a macro average can conceal
a model that has abandoned a class entirely — a failure mode observed during
development.

## Seeds

Five seeds (13, 21, 42, 87, 100) for every system a claim rests on: the
baselines it is compared against, and every system entering a significance test.
Three seeds (13, 42, 100) for context rows that only situate a comparison.

Seed budgets are declared per system in `configs/systems.yaml` **before** runs,
not chosen after seeing results. Standard deviations across seeds are reported
alongside means, and no best-seed number appears anywhere.

Single-run exceptions, labelled as such wherever they appear: the QLoRA
adaptations, the deterministic prompted evaluations, and the cross-lingual
transfer arm (zero-shot, LaBSE, Turkish intermediate training, external test
sets). The tf-idf baseline runs under the full five seeds but its fit is
deterministic, so all five coincide exactly and the tables show no deviation.

## Significance

A hierarchical, cluster-aware paired bootstrap with B = 10,000 replicates. Each
replicate draws:

1. a seed pair (s_A, s_B) uniformly from the available runs of the two systems, and
2. a resample of source groups with replacement, stratified by domain.

The statistic is the paired difference in macro-F1 on the resampled units, so
both seed variance and source clustering enter the interval. Singleton comments
form their own groups, so the resample degenerates to the standard bootstrap
where grouping information is absent.

Family-wise error is controlled with Holm–Bonferroni over the family of systems
in each comparison table, using each system's median p-value across domains.
That median summary is a conservative reporting device, not the basis of any
positive claim: comparisons whose intervals exclude zero are reported
individually, and none is presented as significant after adjustment. As a
robustness check, applying Holm–Bonferroni jointly across all 190 per-domain
one-sided comparisons leaves every adjusted p-value at 1.0 — computed by
`scripts/verify_paper_numbers.py`, so no conclusion depends on how the family is
formed.

A result counts as significant only if its confidence interval excludes zero
**and** it survives adjustment. McNemar's test on the first seed pair is
reported as a distribution-free secondary check, since it is sensitive to
per-instance disagreement rather than to the aggregate metric.

Under this battery, exactly two comparisons in the entire study have intervals
excluding zero, both positive and both about backbone choice:

- mDeBERTa-v3 over XLM-R, LODO Tech: +0.037, CI [+0.005, +0.068]
- XLM-R + DAPT over XLM-R + SupCon, LODO Tech: CI [+0.003, +0.060]

Neither survives Holm adjustment.

## Probes

Beyond accuracy, each run records embedding-geometry and calibration
diagnostics: nearest-neighbour label agreement at k, NMI and ARI against label
clusters, silhouette, anisotropy, TwoNN intrinsic dimension, and expected and
maximum calibration error with Brier score before and after temperature scaling.

These are diagnostic, not decisive. Over 310 runs, NN@10 correlates with
macro-F1 at ρ = 0.79 — but supervised contrastive regularization raises
silhouette from 0.20 to 0.27 and anisotropy from 0.36 to 0.80 while macro-F1
*falls*. Cleaner-looking geometry is not better classification, which is why the
article's claims rest on the metric and the probes are reported as evidence
about mechanism rather than as a proxy for quality.

## Composition matching

Label prior and domain mix agree across gold train, dev and test to within 0.04
percentage points. This is the property whose absence produced the earlier
split's apparent gain: see Section VII of the article and
`results/tables/prior_sweep_lodo.csv`.
