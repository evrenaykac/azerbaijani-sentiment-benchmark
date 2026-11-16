#!/usr/bin/env python3
"""Sensitivity of the quoted per-domain effects to comments without a source identifier and to clustering itself.

    python3 -I singleton_sensitivity.py <runs_dir> <repo> <out_json> [n_boot]

For each quoted comparison: (i) all test items, cluster bootstrap (reproduces the released
statistics), (ii) source-identified items only, (iii) all items with a comment-level
bootstrap (every comment its own unit).
"""
import sys, json, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import *

runs, repo, out = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
NB = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
grp_of, has_of = group_map(repo)
COMPS = [('lodo', 'Tech', 'mdeberta_ft', 'xlmr_ft'), ('lodo', 'Tech', 'xlmr_dapt', 'xlmr_ft'), ('lodo', 'Retail', 'xlmr_dapt_film', 'xlmr_ft'),
         ('lodo', 'Tech', 'mdeberta_dapt', 'xlmr_ft'), ('indomain', 'Social', 'full_mdeberta', 'xlmr_ft'), ('lodo', 'Tech', 'xlmr_dapt', 'xlmr_supcon'),
         ('lodo', 'Tech', 'full', 'mdeberta_ft'), ('lodo', 'Finance', 'full', 'mdeberta_ft'), ('lodo', 'Public', 'full', 'xlmr_ft'),
         ('lodo', 'Social', 'mbert_ft', 'xlmr_ft'), ('lodo', 'Public', 'mbert_ft', 'xlmr_ft'), ('indomain', 'Finance', 'mbert_ft', 'xlmr_ft'), ('indomain', 'Social', 'mbert_ft', 'xlmr_ft')]
res = {}
for regime, scope, A, B in COMPS:
    dA, dB = load_runs(runs, regime, scope, A), load_runs(runs, regime, scope, B)
    base = dA[0]; uall = base['uid'].tolist()
    key = f'{regime}.{scope}.{A}_vs_{B}'; res[key] = {}
    for name in ['all_cluster', 'identified_only', 'all_comment_level']:
        uids = [u for u in uall if has_of[u]] if name == 'identified_only' else uall
        b = base.set_index('uid').loc[uids]
        y = b['gold'].to_numpy(); dom = b['domain'].to_numpy()
        grp = np.array(uids) if name == 'all_comment_level' else np.array([grp_of[u] for u in uids])
        r = both_designs(y, align(dA, uids), align(dB, uids), make_units(dom, grp), n_boot=NB)
        r['n_items'] = len(uids); r['n_units'] = int(len(set(grp)))
        res[key][name] = r
    a, i, c = res[key]['all_cluster'], res[key]['identified_only'], res[key]['all_comment_level']
    print(f'{key:44s} all: {a["diff_mean_over_seeds"]:+.3f} sm{a["seed_mean"]["ci95"]} sp{a["seed_pair"]["ci95"]} n={a["n_items"]} units={a["n_units"]}\n'
          f'{"":44s} id-only: {i["diff_mean_over_seeds"]:+.3f} sm{i["seed_mean"]["ci95"]} sp{i["seed_pair"]["ci95"]} n={i["n_items"]}\n'
          f'{"":44s} comment-level: sm{c["seed_mean"]["ci95"]} sp{c["seed_pair"]["ci95"]}', flush=True)
out.write_text(json.dumps(res, indent=1))
