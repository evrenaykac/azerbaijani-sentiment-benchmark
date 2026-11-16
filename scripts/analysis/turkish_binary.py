#!/usr/bin/env python3
"""Binary (negative vs. positive) reading of the Turkish zero-shot transfer.

    python3 -I turkish_binary.py <runs_dir> <repo> <out_json>

The Turkish-trained model's 3x3 confusion counts are in results/transfer_report.json; per-item
predictions of that run are not in the release, so the binary score is bounded: neutral
predictions on gold negative/positive items counted as errors (lower bound) or as correct
(upper bound). In-language reference: pooled XLM-R fine-tuning (five seeds), same items,
both the three-way prediction and a forced binary decision from the stored logits.
"""
import sys, json, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import *
runs, repo, out = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
tr = json.loads((repo / 'results' / 'transfer_report.json').read_text())
res = {'turkish_training_data': {}}
tot = {'negative': 0, 'neutral': 0, 'positive': 0}
for k, v in tr['turkish_datasets'].items():
    n = v['n']; res['turkish_training_data'][k] = {'id': v['id'], 'n': n, 'share': {l: round(v['labels'][l] / n, 4) for l in tot}}
    for l in tot: tot[l] += v['labels'][l]
N = sum(tot.values()); res['turkish_training_data']['combined'] = {'n': N, 'share': {l: round(tot[l] / N, 4) for l in tot}}
print('Turkish training data:', json.dumps(res['turkish_training_data']))

def binary_from_cm(cm):
    cm = np.array(cm, float)           # rows gold neg/neu/pos, cols predicted
    n_neg, n_pos = cm[0].sum(), cm[2].sum()
    acc_lo = (cm[0, 0] + cm[2, 2]) / (n_neg + n_pos)
    acc_hi = (cm[0, 0] + cm[0, 1] + cm[2, 2] + cm[2, 1]) / (n_neg + n_pos)
    return {'n_items': int(n_neg + n_pos), 'neg_recall': round(cm[0, 0] / n_neg, 4), 'pos_recall': round(cm[2, 2] / n_pos, 4),
            'neg_called_positive': round(cm[0, 2] / n_neg, 4), 'pos_called_negative': round(cm[2, 0] / n_pos, 4),
            'accuracy_lower': round(acc_lo, 4), 'accuracy_upper': round(acc_hi, 4),
            'balanced_accuracy_lower': round((cm[0, 0] / n_neg + cm[2, 2] / n_pos) / 2, 4)}
for key, node in [('xlmr_turkish_only', tr['t1_tr_only_to_az']), ('labse_turkish_only', tr['t2_labse']['tr_only']), ('labse_turkish_plus_az_gold', tr['t2_labse']['tr_plus_az_goldtrain'])]:
    res[key] = binary_from_cm(node['confusion_counts']); res[key]['three_way_macro_f1'] = round(node['macro_f1'], 4)
    res[key]['pred_share'] = [round(float(x), 4) for x in np.array(node['confusion_counts']).sum(0) / np.array(node['confusion_counts']).sum()]
    print(key, res[key])
# in-language reference
accs3, accsb, cms = [], [], []
for s in SEEDS:
    d = pd.read_csv(pathlib.Path(runs) / f'pooled.all.xlmr_ft.s{s}' / 'preds_test.csv')
    cm = np.bincount(d.gold.to_numpy() * 3 + d.pred.to_numpy(), minlength=9).reshape(3, 3); cms.append(cm)
    pol = d[d.gold != 1]
    forced = np.where(pol.logit_pos > pol.logit_neg, 2, 0)
    accsb.append(float((forced == pol.gold.to_numpy()).mean()))
ref = binary_from_cm(np.mean(cms, 0)); ref['forced_binary_accuracy'] = round(float(np.mean(accsb)), 4)
res['xlmr_in_language_pooled_5seeds'] = ref
print('in-language XLM-R (pooled, mean of five seeds):', ref)
out.write_text(json.dumps(res, indent=1))
