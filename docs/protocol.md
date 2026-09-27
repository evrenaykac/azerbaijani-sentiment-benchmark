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
training, not to development, and not to domain-adaptive pretraining. The
pipeline re-verifies this before every fold.

## Metric

Macro-averaged F1 over the three classes, preferred to accuracy because neutral
is both the smallest class and the one systems most often collapse. Per-class F1
is reported for every system in the appendix, since a macro average can conceal
a model that has abandoned a class entirely — a failure mode observed during
development.

## Seeds

Five seeds (13, 21, 42, 87, 100) for the full recipe, its primary baseline
(XLM-R fine-tuning) and its contrastive ablation in both regimes; for the
classical baselines in-domain; for the mid-injection and backbone variants of
the full recipe and mDeBERTa with domain-adaptive pretraining under LODO; and for the pooled encoders. Three seeds for the
remaining systems (13, 21, 42 in-domain; 13, 42, 100 under LODO). The bootstrap draws seed pairs from whatever runs exist,
and the number of pairs is recorded per comparison in `results/stats/`.

Seed budgets are declared per system in `configs/systems.yaml` **before** runs,
not chosen after seeing results. Standard deviations across seeds are reported
alongside means, and no best-seed number appears anywhere.

Single-run exceptions, labeled as such wherever they appear: the QLoRA
adaptations, the deterministic prompted evaluations, and the cross-lingual
transfer arm (zero-shot, LaBSE, Turkish intermediate training, external
datasets). The tf-idf baseline runs under the full five seeds but its fit is
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

For compact exploratory reporting, each system's five domain-specific p-values
are summarized by their median and Holm-adjusted across the systems of each
comparison table. That summary is not a formal combination test and is not used
for confirmatory inference: comparisons whose intervals exclude zero are
reported individually, and none is presented as significant after adjustment. As a
robustness check, applying Holm–Bonferroni jointly across all 190 per-domain
one-sided comparisons leaves every adjusted p-value at 1.0 — computed by
`scripts/verify_paper_numbers.py`, so the null result for the adaptation
techniques does not depend on how the family is formed.

The article calls a positive effect statistically supported when its
prespecified domain-specific 95% interval excludes zero; none of those effects
also survives Holm adjustment. McNemar's test on the first seed pair is
reported as a distribution-free secondary check, since it is sensitive to
per-instance disagreement rather than to the aggregate metric.

Under this battery, exactly two comparisons on the audited benchmark have positive intervals excluding zero:

- mDeBERTa-v3 over XLM-R, LODO Tech: +0.037, CI [+0.005, +0.068] (backbone choice)
- XLM-R + DAPT over XLM-R + SupCon, LODO Tech: CI [+0.003, +0.060] (domain-adaptive
  pretraining against contrastive regularization; neither is significantly better than plain XLM-R fine-tuning)

Neither survives Holm adjustment.

## Probes

Beyond accuracy, each in-domain and LODO encoder run records
embedding-geometry and calibration diagnostics: nearest-neighbor label agreement at k, NMI and ARI against label
clusters, silhouette, anisotropy, TwoNN intrinsic dimension, and expected and
maximum calibration error with Brier score before and after temperature scaling.

These are diagnostic, not decisive. Over the 310 LODO encoder runs, NN@10 correlates with
macro-F1 at ρ = 0.79 — but along the in-domain ablation chain, adding FiLM injection and supervised contrastive regularization on top of DAPT raises silhouette from 0.20 to 0.27 and anisotropy from 0.36 to 0.80 while macro-F1
*falls*. Cleaner-looking geometry is not better classification, which is why the
article's claims rest on the metric and the probes are reported as evidence
about mechanism rather than as a proxy for quality.

## Composition matching

Label prior and domain mix agree across gold train, dev and test to within 0.04
percentage points. The earlier split lacked this property; Section VII of the
article shows that the mismatch masked rather than created that split's
apparent gain, which came from the teacher-generated bulk labels of the earlier
protocol. See `results/tables/prior_sweep_lodo.csv` and
`results/earlier_split/`.
