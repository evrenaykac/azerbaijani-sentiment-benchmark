#!/usr/bin/env python3
"""System averages on the whole test partition and on a subset of it, at full precision.

    python3 -I cohort_tables.py <runs_dir> <repo> <cohort.csv> <out_json> [subset_value]

<cohort.csv> has the columns uid,cohort; the subset is the rows whose cohort equals
<subset_value> (default: blind).
"""
import sys, json, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import macro_f1, per_class_f1, DOMS, SEEDS
runs, repo, cohort_csv, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], pathlib.Path(sys.argv[4])
coh = dict(pd.read_csv(cohort_csv, dtype=str).values)
SUB = sys.argv[5] if len(sys.argv) > 5 else 'blind'          # value of the cohort column that defines the subset
OTHER = [v for v in sorted(set(coh.values())) if v != SUB]
def load(regime, scope, system):
    o = []
    for s in SEEDS:
        f = runs / f'{regime}.{scope}.{system}.s{s}' / 'preds_test.csv'
        if f.exists(): o.append(pd.read_csv(f, usecols=['uid', 'domain', 'gold', 'pred'], dtype={'uid': str}))
    return o
def sc(dfs, which):
    v = []; pc = []
    for d in dfs:
        m = np.ones(len(d), bool) if which == 'all' else (d['uid'].map(coh) == which).to_numpy()
        y, p = d['gold'].to_numpy()[m], d['pred'].to_numpy()[m]
        v.append(macro_f1(y, p)); pc.append(per_class_f1(y, p))
    return float(np.mean(v)), float(np.std(v)), np.mean(pc, axis=0).tolist(), len(v)
res = {'indomain': {}, 'lodo': {}, 'pooled': {}}
for regime in ('indomain', 'lodo'):
    systems = sorted({p.name.split('.')[2] for p in runs.glob(f'{regime}.*')})
    for system in systems:
        e = {}
        for which in ('all', SUB):
            per = {}
            for dom in DOMS:
                dfs = load(regime, dom, system)
                if not dfs: continue
                m, sd, pc, k = sc(dfs, which); per[dom] = {'mean': m, 'sd': sd, 'n_seeds': k}
            if len(per) == 5:
                e[which] = {'avg': float(np.mean([per[d]['mean'] for d in DOMS])), 'per_domain': per}
        if e: res[regime][system] = e
for system in ('xlmr_ft', 'mdeberta_ft', 'full'):
    dfs = load('pooled', 'all', system); e = {}
    for which in ['all', SUB] + OTHER:
        m, sd, pc, k = sc(dfs, which); e[which] = {'mean': m, 'sd': sd, 'per_class_f1': pc, 'n_seeds': k}
    res['pooled'][system] = e
for name, f in [('qlora_llama31', 'llm/llama31/preds_test.csv'), ('qlora_qwen25', 'llm/qwen25/preds_test.csv'), ('qlora_qwen3', 'llm/qwen3/preds_test.csv'),
                ('gpt4o_3shot', 'llm/openai/preds_gpt4o_3shot.csv'), ('gpt4o_0shot', 'llm/openai/preds_gpt4o_0shot.csv'),
                ('gpt4o_mini_3shot', 'llm/openai/preds_gpt4o_mini_3shot.csv'), ('gpt4o_mini_0shot', 'llm/openai/preds_gpt4o_mini_0shot.csv')]:
    d = pd.read_csv(repo / 'results' / f, dtype={'uid': str}); e = {}
    for which in ['all', SUB] + OTHER:
        m, sd, pc, k = sc([d], which); e[which] = {'mean': m, 'per_class_f1': pc, 'n_seeds': 1}
    res['pooled'][name] = e
out.write_text(json.dumps(res, indent=1))
for regime in ('indomain', 'lodo'):
    print(regime)
    for k, v in sorted(res[regime].items(), key=lambda kv: -kv[1]['all']['avg']):
        print(f"  {k:18s} all {v['all']['avg']:.5f}  blind {v[SUB]['avg']:.5f}  diff_to_xlmr all {100*(v['all']['avg']-res[regime]['xlmr_ft']['all']['avg']):+.2f} blind {100*(v[SUB]['avg']-res[regime]['xlmr_ft'][SUB]['avg']):+.2f}")
print('pooled')
for k, v in res['pooled'].items(): print(f"  {k:18s} all {v['all']['mean']:.5f} blind {v[SUB]['mean']:.5f} verif {v[OTHER[0]]['mean']:.5f}  blind per-class {np.round(v[SUB]['per_class_f1'],3).tolist()}  all per-class {np.round(v['all']['per_class_f1'],3).tolist()}")
