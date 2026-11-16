#!/usr/bin/env python3
"""Consistency checks on quoted figures, from the released ledger and stored predictions.

    python3 -I consistency_checks.py <runs_dir> <repo> <out_json>
"""
import sys, json, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import *
runs, repo, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
res = {}
led = pd.read_csv(repo / 'results' / 'runs_master.csv')
led['macro_f1'] = pd.to_numeric(led['macro_f1'], errors='coerce')
# DAPT gain in the XLM-R averages, from unrounded means
def avg(regime, system):
    s = led[(led.regime == regime) & (led.system == system)]
    return float(s.groupby('scope')['macro_f1'].mean().mean())
for regime in ['indomain', 'lodo']:
    a, b = avg(regime, 'xlmr_dapt'), avg(regime, 'xlmr_ft')
    res[f'dapt_gain_{regime}'] = {'xlmr_dapt': round(a, 5), 'xlmr_ft': round(b, 5), 'diff_points_unrounded': round(100 * (a - b), 3), 'diff_points_from_rounded': round(100 * (round(a, 3) - round(b, 3)), 1)}
    print(regime, res[f'dapt_gain_{regime}'])
# the LODO chain values
for sname in ['xlmr_ft', 'xlmr_dapt', 'xlmr_film', 'xlmr_dapt_film', 'full', 'mdeberta_ft', 'mdeberta_dapt', 'full_mdeberta']:
    print(f'  LODO avg {sname:16s} {avg("lodo", sname):.4f}   in-domain avg {avg("indomain", sname) if len(led[(led.regime=="indomain")&(led.system==sname)]) else float("nan"):.4f}')
res['lodo_chain'] = {s: round(avg('lodo', s), 4) for s in ['xlmr_ft', 'xlmr_dapt', 'xlmr_film', 'xlmr_dapt_film', 'full', 'mdeberta_ft', 'mdeberta_dapt', 'full_mdeberta']}
# best vs worst encoder under LODO
enc = {s: avg('lodo', s) for s in ['mbert_ft', 'allma_ft', 'xlmr_ft', 'mdeberta_ft']}
res['lodo_encoders'] = {k: round(v, 4) for k, v in enc.items()}; print('LODO plain encoders:', res['lodo_encoders'])
# any run predicting (almost) a single class? every stored run of the released ledger
rows = []
for d in sorted(runs.iterdir()):
    f = d / 'preds_test.csv'
    if not f.exists(): continue
    df = pd.read_csv(f, usecols=['gold', 'pred'])
    share = np.bincount(df.pred.to_numpy(), minlength=3) / len(df)
    gshare = np.bincount(df.gold.to_numpy(), minlength=3) / len(df)
    m = json.loads((d / 'metrics.json').read_text()) if (d / 'metrics.json').exists() else {}
    rows.append({'run': d.name, 'system': d.name.split('.')[2], 'min_pred_share': float(share.min()), 'max_pred_share': float(share.max()),
                 'max_abs_dev_from_gold': float(np.abs(share - gshare).max()), 'macro_f1': macro_f1(df.gold.to_numpy(), df.pred.to_numpy()),
                 'backbone': m.get('backbone_resolved', ''), 'dtype': str(m.get('env', {}).get('backbone_dtype', m.get('backbone_dtype', '')))})
R = pd.DataFrame(rows)
print('stored runs:', len(R), '| smallest predicted-class share over all runs:', round(R.min_pred_share.min(), 4), '| largest:', round(R.max_pred_share.max(), 4))
print(R.sort_values('min_pred_share').head(6)[['run', 'min_pred_share', 'max_pred_share', 'macro_f1']].to_string())
md = R[R.system.str.contains('mdeberta')]
print('mDeBERTa-backbone runs:', len(md), '| smallest predicted-class share:', round(md.min_pred_share.min(), 4), '| lowest macro-F1:', round(md.macro_f1.min(), 4))
res['single_class_check'] = {'n_runs': int(len(R)), 'min_pred_share_any_run': round(float(R.min_pred_share.min()), 4), 'max_pred_share_any_run': round(float(R.max_pred_share.max()), 4),
                             'n_mdeberta_runs': int(len(md)), 'min_pred_share_mdeberta': round(float(md.min_pred_share.min()), 4), 'min_macro_f1_mdeberta': round(float(md.macro_f1.min()), 4),
                             'lowest_runs': R.sort_values('min_pred_share').head(5)[['run', 'min_pred_share', 'macro_f1']].round(4).to_dict('records')}
m = json.loads((runs / 'lodo.Tech.mdeberta_ft.s13' / 'metrics.json').read_text())
print('metrics keys with dtype info:', {k: v for k, v in m.items() if 'dtype' in k.lower() or 'precision' in k.lower()}, '| env:', m.get('env'))
out.write_text(json.dumps(res, indent=1))
