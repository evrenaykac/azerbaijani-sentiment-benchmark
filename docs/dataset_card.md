# Dataset card — AzSentBench

## Overview

| | |
|---|---|
| **Name** | AzSentBench (corpus and benchmark) |
| **Language** | Azerbaijani (`az`) |
| **Task** | Comment-level sentiment classification, three classes |
| **Size** | 271,278 comments; 10,000 gold, 261,278 bulk |
| **Domains** | Tech, Finance, Social, Retail, Public Services |
| **License** | CC BY 4.0 (compiled dataset); text not redistributed |

## Provenance

Publicly posted comment sections of consumer-facing online platforms. No author-level attributes
were collected; source pages appear solely as salted hashes. Collection covered
five domains chosen to span topics, from consumer complaints to public service
discussion; all share an informal register.

## Labels

Three classes: negative, neutral, positive.

Two supervision layers, reported separately because their protocols differ:

- **Gold** (10,000): the partition used for all reported evaluation. 4,338 carry
  three annotations plus adjudication (1,257 of them from the verification
  batch, with the earlier label visible); the rest carry a single
  annotator's label. The 3,000-item **test** partition is fully triple-annotated
  and adjudicated.
- **Bulk** (261,278): 87,069 labeled in a wide single-annotator pass (6,912 of
  which also carry triple annotation) and 174,209 unlabeled. The labeled bulk comments make up most of every
  supervised training pool (about 94% when all domains are pooled); bulk text,
  with labels discarded, also serves domain-adaptive pretraining.

## Annotation

Three native-speaker annotators labeling from a written guideline (blind in the
10,000-item batch; with the earlier label visible in the 1,257-item verification
batch), and a fourth team member acting as adjudicator who reviewed every item
and issued the final label. Agreement statistics exclude the adjudicator.

Fleiss' κ = 0.756 on the blind 10,000-item batch (substantial on the
Landis–Koch scale). Pairwise Cohen's κ ranges 0.750–0.762 with no annotator
behaving as an outlier, and the three annotators' marginal label distributions
agree to within 0.5 percentage points. Per-domain κ runs from 0.685 (Finance,
where sentiment and factual reporting mix) to 0.783 (Public Services).

See [`../data/README.md`](../data/README.md) on the two batches and why their
statistics are not pooled.

## Splits

Source-grouped, composition-matched, and audited against four leakage
invariants — all zero, all re-derivable by `scripts/verify_invariants.py`. See
[`protocol.md`](protocol.md).

## Intended use

Evaluating sentiment models for Azerbaijani, and evaluating *evaluations*: the
release includes the audit tooling and stored predictions so that split quality
itself can be inspected, which is the point of the accompanying article.

## Out-of-scope use

Not suitable for author-level inference, demographic inference, or any use
requiring identification of the people who wrote the comments — the release
carries no author information, by design. The corpus reflects the user populations
of the platforms it was collected from and should not be treated as
representative of Azerbaijani speakers generally.

## Known limitations

Listed candidly in [`../data/README.md`](../data/README.md): 71.0% source-group
identifier coverage, 89.2% of gold-train carrying a single annotator's label,
two annotation protocols, and neutral as the hardest class on average.

## Maintenance

Issues and corrections through the GitHub issue tracker. Corrections to labels
or splits will be released as versioned updates rather than silent edits, since
silent edits to a benchmark are exactly the failure mode this project documents.
