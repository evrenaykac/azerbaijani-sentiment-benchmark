#!/usr/bin/env python3
"""How pooled systems behave on the two annotation cohorts of the test partition.

    python3 -I cohort_behaviour.py <runs_dir> <repo> <test_cohort.csv> <out_json>

For each pooled system: share of comments predicted neutral, accuracy, and per-class F1,
separately on the blind cohort (1,743) and the verification cohort (1,257).
"""
import sys, json, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import *
runs, repo, cohort_csv, out = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3], pathlib.Path(sys.argv[4])
coh = pd.read_csv(cohort_csv, dtype=str).set_index('uid')['cohort']
def llm(path):
    return [pd.read_csv(repo / 'results' / path, usecols=['uid', 'domain', 'gold', 'pred'], dtype={'uid': str})]
SYS = {'xlmr_ft': load_runs(runs, 'pooled', 'all', 'xlmr_ft'), 'mdeberta_ft': load_runs(runs, 'pooled', 'all', 'mdeberta_ft'),
       'full': load_runs(runs, 'pooled', 'all', 'full'),
       'qlora_llama31': llm('llm/llama31/preds_test.csv'), 'qlora_qwen25': llm('llm/qwen25/preds_test.csv'), 'qlora_qwen3': llm('llm/qwen3/preds_test.csv'),
       'gpt4o_3shot': llm('llm/openai/preds_gpt4o_3shot.csv'), 'gpt4o_0shot': llm('llm/openai/preds_gpt4o_0shot.csv'),
       'gpt4o_mini_3shot': llm('llm/openai/preds_gpt4o_mini_3shot.csv'), 'gpt4o_mini_0shot': llm('llm/openai/preds_gpt4o_mini_0shot.csv')}
res = {}
for name, dfs in SYS.items():
    base = dfs[0]; uids = base['uid'].tolist(); y = base['gold'].to_numpy(); P = align(dfs, uids)
    c = coh.loc[uids].to_numpy(); row = {'n_runs': len(dfs)}
    for part, m in (('all', np.ones(len(y), bool)), ('blind', c == 'blind'), ('verification', c == 'verification')):
        yy = y[m]
        row[part] = {'n': int(m.sum()), 'gold_share': [round(float((yy == k).mean()), 4) for k in range(3)],
                     'pred_share': [round(float(np.mean([(p[m] == k).mean() for p in P])), 4) for k in range(3)],
                     'accuracy': round(float(np.mean([(p[m] == yy).mean() for p in P])), 4),
                     'macro_f1': round(float(np.mean([macro_f1(yy, p[m]) for p in P])), 4),
                     'per_class_f1': [round(float(v), 4) for v in np.mean([per_class_f1(yy, p[m]) for p in P], axis=0)]}
        if part == 'verification':
            nn = yy != 1  # negative/positive gold only
            row[part]['negpos_called_neutral'] = round(float(np.mean([(p[m][nn] == 1).mean() for p in P])), 4)
            row[part]['negpos_polarity_flipped'] = round(float(np.mean([((p[m][nn] != 1) & (p[m][nn] != yy[nn])).mean() for p in P])), 4)
        if part == 'blind':
            nn = yy != 1
            row[part]['negpos_called_neutral'] = round(float(np.mean([(p[m][nn] == 1).mean() for p in P])), 4)
            row[part]['negpos_polarity_flipped'] = round(float(np.mean([((p[m][nn] != 1) & (p[m][nn] != yy[nn])).mean() for p in P])), 4)
    res[name] = row
    print(f"{name:18s} runs={len(dfs)}  ALL F1 {row['all']['macro_f1']:.3f} acc {row['all']['accuracy']:.3f} | BLIND F1 {row['blind']['macro_f1']:.3f} acc {row['blind']['accuracy']:.3f} predNEU {row['blind']['pred_share'][1]:.3f} perclass {row['blind']['per_class_f1']} n/p->neu {row['blind']['negpos_called_neutral']:.3f} flip {row['blind']['negpos_polarity_flipped']:.3f} | VERIF acc {row['verification']['accuracy']:.3f} predNEU {row['verification']['pred_share'][1]:.3f} n/p->neu {row['verification']['negpos_called_neutral']:.3f} flip {row['verification']['negpos_polarity_flipped']:.3f}")
out.write_text(json.dumps(res, indent=1))
