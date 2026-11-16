#!/usr/bin/env python3
"""Results on the blind and the verification cohort of the test partition.

    python3 -I cohort_sensitivity.py <runs_dir> <repo> <test_cohort.csv> <out_json>
"""
import sys, json, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import *

runs, repo, cohort_csv, out = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3], pathlib.Path(sys.argv[4])
coh = dict(pd.read_csv(cohort_csv, dtype=str).values)
grp_of, _ = group_map(repo)
res = {'pooled': {}, 'lodo': {}, 'indomain': {}, 'llm': {}}

def cohort_scores(dfs):
    rows = {}
    for name in ['all', 'blind', 'verification']:
        vals = []
        for d in dfs:
            m = np.ones(len(d), bool) if name == 'all' else (d['uid'].map(coh) == name).to_numpy()
            vals.append(macro_f1(d['gold'].to_numpy()[m], d['pred'].to_numpy()[m]))
        rows[name] = round(float(np.mean(vals)), 4)
    return rows

# pooled encoders (five seeds) and the language models (single runs)
for sysname in ['mdeberta_ft', 'xlmr_ft', 'full']:
    res['pooled'][sysname] = cohort_scores(load_runs(runs, 'pooled', 'all', sysname))
for name, f in [('qlora_llama31', 'llm/llama31/preds_test.csv'), ('qlora_qwen25', 'llm/qwen25/preds_test.csv'), ('qlora_qwen3', 'llm/qwen3/preds_test.csv'),
                ('gpt4o_3shot', 'llm/openai/preds_gpt4o_3shot.csv'), ('gpt4o_0shot', 'llm/openai/preds_gpt4o_0shot.csv'),
                ('gpt4o_mini_3shot', 'llm/openai/preds_gpt4o_mini_3shot.csv'), ('gpt4o_mini_0shot', 'llm/openai/preds_gpt4o_mini_0shot.csv')]:
    d = pd.read_csv(repo / 'results' / f, dtype={'uid': str})
    res['llm'][name] = cohort_scores([d])
print('POOLED (macro-F1: all / blind 1,743 / verification 1,257)')
for k, v in {**res['pooled'], **res['llm']}.items():
    print(f'  {k:18s} {v["all"]:.3f}  {v["blind"]:.3f}  {v["verification"]:.3f}')

# LODO and in-domain averages over domains, per cohort
for regime in ['lodo', 'indomain']:
    systems = sorted({p.name.split('.')[2] for p in pathlib.Path(runs).glob(f'{regime}.*')})
    for sysname in systems:
        per = {c: [] for c in ['all', 'blind', 'verification']}
        for dom in DOMS:
            sc = cohort_scores(load_runs(runs, regime, dom, sysname))
            for c in per: per[c].append(sc[c])
        res[regime][sysname] = {c: round(float(np.mean(v)), 4) for c, v in per.items()}
        res[regime][sysname]['per_domain_blind'] = dict(zip(DOMS, per['blind']))
    print(f'{regime.upper()} average over domains (all / blind / verification)')
    for k, v in sorted(res[regime].items(), key=lambda kv: -kv[1]['all']):
        print(f'  {k:18s} {v["all"]:.3f}  {v["blind"]:.3f}  {v["verification"]:.3f}')
    a = [v['all'] for v in res[regime].values()]; b = [v['blind'] for v in res[regime].values()]
    from scipy.stats import spearmanr
    rho = spearmanr(a, b)[0]
    res[regime + '_rank_correlation_all_vs_blind'] = round(float(rho), 4)
    print(f'  Spearman rho between the all-items and blind-only system averages: {rho:.3f}')

# key pooled differences on the blind cohort only (both designs)
res['pooled_blind_differences'] = {}
for A, B in [('mdeberta_ft', 'xlmr_ft'), ('full', 'xlmr_ft')]:
    dA, dB = load_runs(runs, 'pooled', 'all', A), load_runs(runs, 'pooled', 'all', B)
    base = dA[0]; uids = [u for u in base['uid'] if coh[u] == 'blind']
    y = base.set_index('uid').loc[uids, 'gold'].to_numpy(); dom = base.set_index('uid').loc[uids, 'domain'].to_numpy()
    grp = np.array([grp_of[u] for u in uids])
    r = both_designs(y, align(dA, uids), align(dB, uids), make_units(dom, grp), n_boot=10000)
    res['pooled_blind_differences'][f'{A}_vs_{B}'] = r
    uall = base['uid'].tolist(); ya = base['gold'].to_numpy(); doma = base['domain'].to_numpy(); grpa = np.array([grp_of[u] for u in uall])
    r_all = both_designs(ya, align(dA, uall), align(dB, uall), make_units(doma, grpa), n_boot=10000)
    res['pooled_blind_differences'][f'{A}_vs_{B}_all_items'] = r_all
    print(f'pooled {A} - {B}: all items {r_all["diff_mean_over_seeds"]:+.4f} seed-mean {r_all["seed_mean"]["ci95"]} | blind only {r["diff_mean_over_seeds"]:+.4f} seed-mean {r["seed_mean"]["ci95"]} seed-pair {r["seed_pair"]["ci95"]}')
out.write_text(json.dumps(res, indent=1))
