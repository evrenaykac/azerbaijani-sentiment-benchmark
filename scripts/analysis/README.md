# Analyses computed from the stored predictions and the annotation records

These scripts produce the files in `results/analysis/`. None of them needs the comment
text or a GPU. They read the stored per-item predictions of the reported runs, which
are attached to release v1.0.0 as three archives, and the files of this repository.

| Argument | What to pass |
|---|---|
| `<runs_dir>` | the `runs/` folder extracted from `predictions.tar.gz` |
| `<earlier_split_runs_dir>` | the `runs/` folder extracted from `predictions_earlier_split.tar.gz` |
| `<label_control_runs_dir>` | the `runs/` folder extracted from `predictions_label_controls.tar.gz` |
| `<repo>` | the root of this repository |
| `<cohort.csv>` | a file with the columns `uid,cohort`, such as `results/analysis/subsets/test_cohort.csv` |

| Script | What it computes | Used in |
|---|---|---|
| `agreement.py` | Agreement on the test partition and per batch, with bootstrap intervals | Section III-C, Tables 2-4 |
| `cohort_tables.py` | System averages on the whole test partition and on a subset of it, at full precision | Tables 7 and 19, Section VI-F |
| `cohort_sensitivity.py` | Results on the blind and the verification cohort of the test partition | Section VI-F |
| `cohort_behaviour.py` | How pooled systems behave on the two annotation cohorts of the test partition | Sections VI-B and VI-F |
| `subset_statistics.py` | Every released per-domain comparison, recomputed on a subset of the test partition | Section VI-F |
| `count_effects.py` | Count per-domain effects by level of support in a directory of stats_*.json files | Sections VI-A and VI-F |
| `two_sided_statistics.py` | One convention for intervals and p-values, from the released statistics | Sections IV-C and VI-A |
| `singleton_sensitivity.py` | Sensitivity of the quoted per-domain effects to comments without a source identifier and to clustering itself | Section VI-F |
| `encoder_vs_llm.py` | Test-sampling uncertainty of the encoder-versus-LLM comparisons | Section VI-B |
| `turkish_binary.py` | Binary (negative vs. positive) reading of the Turkish zero-shot transfer | Section VI-C |
| `stage2_evidence.py` | Direct evidence on the Stage-2 baseline, from stored predictions only | Section VII |
| `baseline_evidence.py` | Direct evidence on the degraded Stage-2 baseline | Section VII, Table 13 |
| `teacher_label_shares.py` | Label composition of the pre-audit training pools, by held-out target | Section VII, Table 14 |
| `consistency_checks.py` | Consistency checks on quoted figures, from the released ledger and stored predictions | Sections III-D, IV-E and VI-A |

## Usage

```bash
python3 -I scripts/analysis/agreement.py <benchmark_repo> <test_cohort.csv> <out_json> [n_boot]
python3 -I scripts/analysis/cohort_tables.py <runs_dir> <repo> <cohort.csv> <out_json> [subset_value]
python3 -I scripts/analysis/cohort_sensitivity.py <runs_dir> <repo> <test_cohort.csv> <out_json>
python3 -I scripts/analysis/cohort_behaviour.py <runs_dir> <repo> <test_cohort.csv> <out_json>
python3 -I scripts/analysis/subset_statistics.py <runs_dir> <repo> <subset_uids.txt|ALL> <out_dir> [family ...]
python3 -I scripts/analysis/count_effects.py <stats_dir> [<out_json>]
python3 -I scripts/analysis/two_sided_statistics.py <repo> <out_json> [<out_csv>]
python3 -I scripts/analysis/singleton_sensitivity.py <runs_dir> <repo> <out_json> [n_boot]
python3 -I scripts/analysis/encoder_vs_llm.py <runs_dir> <repo> <out_json> [n_boot]
python3 -I scripts/analysis/turkish_binary.py <runs_dir> <repo> <out_json>
python3 -I scripts/analysis/stage2_evidence.py <earlier_split_runs_dir> <out_json>
python3 -I scripts/analysis/baseline_evidence.py <earlier_split_runs> <label_control_runs> <out_json>
python3 -I scripts/analysis/teacher_label_shares.py <earlier_split_runs_dir> <out_json> [<label_control_runs_dir>]
python3 -I scripts/analysis/consistency_checks.py <runs_dir> <repo> <out_json>
```

## Subsets of the test partition

`results/analysis/subsets/` holds the three files that define the subsets:
`test_cohort.csv` (blind or verification cohort of every test comment),
`blind_cohort_uids.txt` (the 1,743 comments of the blind cohort) and
`without_near_duplicates_uids.txt` (the 2,841 test comments that have no labeled
training comment at a character 5-gram Jaccard similarity of 0.70 or above;
`test_near_duplicate_flags.csv` gives the same information per comment).

`subset_statistics.py` recomputes the 190 per-domain comparisons under both bootstrap
designs on a subset; its outputs for the two subsets are in
`results/analysis/statistics_blind_cohort/` and
`results/analysis/statistics_without_near_duplicates/`.

## Near-duplicate check

`tools/near_duplicates.py` compares every development and test comment with every
training comment and needs the comment text (see `docs/data_access.md`). Its output,
which contains identifiers and similarities only, is in `results/near_duplicates/`.
