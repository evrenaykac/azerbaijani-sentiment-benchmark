# Label and data-strategy controls (Section VII, Table 15)

Thirty-six training runs on the audited split that separate the two things
the Stage 2 → 2b step of the article changes at once: the labels of the
teacher-labeled bulk comments, and their contribution to the size and
composition of the training pool. All runs use the released pipeline
(`src/azsent`), three seeds (13, 42, 100), and the two systems the Section VII
ladder compares: plain XLM-R fine-tuning (`xlmr_ft`) and the full adaptation
recipe (`full`). They were trained on one RTX 3090 on 9 October 2026 with the
released configuration; the DAPT checkpoints were retrained on the same
machine, which is why the human-label condition was re-run there as well.

| condition | run tag | pool | what differs from the released strategy |
|---|---|---|---|
| H | `human` | gold-train + human-labeled bulk comments (`--pool-mode gold+bulk`) | nothing; re-run in the control environment |
| T | `relabel` | the same comments, bulk labels replaced by the scope teacher's labels (`--pool-mode gold+bulk_relabeled`) | labels only: the teacher's labels disagree with the human labels on 31.3% (pooled) and 32.0% (Public fold) of the bulk rows |
| S | `silver` | H plus every unlabeled bulk comment with a teacher label (`--pool-mode gold+bulk --silver-frac 1.0`) | the Stage-2 data strategy: pool grows from 87,589 to 253,140 (pooled) and from 79,788 to 208,340 (Public fold) |

The teacher is the one of Stage 2: XLM-R trained on gold-train of the scope
(`python -m azsent.teacher --scope all` / `--scope lodo_Public`), which labels
every bulk comment of the scope.

Files:

- `label_controls.json` — seed means, per-domain means, pool composition and
  class prior per condition and system; the full-minus-xlmr_ft difference per
  condition and the T−H and S−H drops per system, each under the two bootstrap
  designs of Section IV-C; the change of the recipe's gap relative to H under
  a shared source-group resample. Produced by `scripts/label_controls.py` from
  the released predictions (`predictions_label_controls.tar.gz` of the GitHub
  release) and `data/corpus_index.csv.gz`.
- `runs_label_controls.csv` — one ledger row per run (macro-F1, per-class F1,
  pool composition, DAPT checkpoint, steps, wall time, GPU). These runs are not
  part of `results/runs_master.csv`, whose 616 runs remain the benchmark
  results; no benchmark number uses teacher labels.
- `teacher_vs_human.json` — the exact confusion between human and teacher
  labels on the bulk rows that enter each pool (`teacher_human_disagreement`),
  per domain, and the teachers' quality reports; computed on the training machine because it needs the
  prepared pools rather than the predictions.

The archive holds, per run, `metrics.json`, `preds_test.csv`, `preds_dev.csv`
and `DONE.json` (no embeddings: the controls use no probe), plus the teachers'
quality reports, the DAPT metrics and the teacher-label files.

Reproducing the numbers of Table 15 and the surrounding text:

```bash
mkdir -p work_lc && tar xzf predictions_label_controls.tar.gz -C work_lc
python scripts/label_controls.py --runs work_lc/runs/runs --out work_lc/out
```

About three minutes on one CPU core; the output matches this folder to the
reported precision.
