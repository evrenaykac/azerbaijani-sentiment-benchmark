#!/usr/bin/env python3
"""Direct evidence on the degraded Stage-2 baseline.

    python3 -I baseline_evidence.py <earlier_split_runs> <label_control_runs> <out_json>

For plain XLM-R fine-tuning and the full recipe on the Public Services target, under each
training-data condition: per-seed macro-F1, per-class F1, predicted class shares on the target
test set and on the source-domain development set, label shares of the training pool, and the
development macro-F1 of the selected checkpoint.
"""
import sys, json, pathlib
import numpy as np, pandas as pd
es, lc, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
def f1s(y, p):
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3).astype(float)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp; den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
COND = [('pre-audit split, teacher-labeled pool (Stage 2)', es, ''), ('pre-audit split, human labels (Stage 2b)', es, '.frac000'),
        ('audited split, H', lc, '.human'), ('audited split, T', lc, '.relabel'), ('audited split, S', lc, '.silver')]
res = {}
for name, root, tag in COND:
    for system in ('xlmr_ft', 'full'):
        seeds = []
        for s in (13, 21, 42, 87, 100):
            d = root / f'lodo.Public.{system}.s{s}{tag}'
            if not (d / 'preds_test.csv').exists(): continue
            m = json.loads((d / 'metrics.json').read_text()); t = pd.read_csv(d / 'preds_test.csv')
            y, p = t['gold'].to_numpy(), t['pred'].to_numpy(); f = f1s(y, p)
            comp = m['composition']['train_by_domain_label']; tot = {l: sum(v for k, v in comp.items() if k.endswith('|' + l)) for l in ('negative', 'neutral', 'positive')}
            n = sum(tot.values())
            rec = {'seed': s, 'macro_f1': round(float(f.mean()), 4), 'f1': [round(float(v), 3) for v in f],
                   'pred_share': [round(float((p == k).mean()), 3) for k in range(3)], 'gold_share': [round(float((y == k).mean()), 3) for k in range(3)],
                   'train_share': [round(tot[l] / n, 3) for l in ('negative', 'neutral', 'positive')], 'train_total': int(m['composition']['train_total']),
                   'teacher_labeled': int(m['composition'].get('bulk_silver', 0)), 'dev_best_macro_f1': round(float(m['dev_best_macro_f1']), 4),
                   'steps_run': m['hyperparams'].get('steps_run'), 'n_test': int(len(y)), 'smallest_predicted_class_share': round(float(min((p == k).mean() for k in range(3))), 3)}
            if (d / 'preds_dev.csv').exists():
                dv = pd.read_csv(d / 'preds_dev.csv'); yd, pd_ = dv['gold'].to_numpy(), dv['pred'].to_numpy(); fd = f1s(yd, pd_)
                rec['dev'] = {'n': int(len(yd)), 'macro_f1': round(float(fd.mean()), 4), 'f1': [round(float(v), 3) for v in fd],
                              'pred_share': [round(float((pd_ == k).mean()), 3) for k in range(3)], 'gold_share': [round(float((yd == k).mean()), 3) for k in range(3)]}
            seeds.append(rec)
        if not seeds: continue
        agg = {'n_seeds': len(seeds), 'macro_f1_by_seed': [r['macro_f1'] for r in seeds], 'mean_macro_f1': round(float(np.mean([r['macro_f1'] for r in seeds])), 4),
               'mean_f1': [round(float(np.mean([r['f1'][k] for r in seeds])), 3) for k in range(3)],
               'mean_pred_share': [round(float(np.mean([r['pred_share'][k] for r in seeds])), 3) for k in range(3)],
               'gold_share': seeds[0]['gold_share'], 'train_share': seeds[0]['train_share'], 'train_total': seeds[0]['train_total'], 'teacher_labeled': seeds[0]['teacher_labeled'],
               'dev_best_macro_f1_by_seed': [r['dev_best_macro_f1'] for r in seeds], 'n_test': seeds[0]['n_test'],
               'smallest_predicted_class_share_any_seed': min(r['smallest_predicted_class_share'] for r in seeds)}
        if 'dev' in seeds[0]:
            agg['dev_mean_f1'] = [round(float(np.mean([r['dev']['f1'][k] for r in seeds])), 3) for k in range(3)]
            agg['dev_mean_pred_share'] = [round(float(np.mean([r['dev']['pred_share'][k] for r in seeds])), 3) for k in range(3)]
            agg['dev_gold_share'] = seeds[0]['dev']['gold_share']; agg['dev_n'] = seeds[0]['dev']['n']
        res[f'{name} | {system}'] = {'summary': agg, 'seeds': seeds}
        a = agg
        print(f"{name:52s} {system:8s} F1 {a['mean_macro_f1']:.3f} seeds {a['macro_f1_by_seed']}  classF1 {a['mean_f1']}  pred {a['mean_pred_share']} gold {a['gold_share']}  train {a['train_share']} (n={a['train_total']}, teacher {a['teacher_labeled']})  dev {a['dev_best_macro_f1_by_seed']}" + (f"  devF1 {a['dev_mean_f1']} devpred {a['dev_mean_pred_share']} devgold {a['dev_gold_share']}" if 'dev_mean_f1' in a else ''))
out.write_text(json.dumps(res, indent=1))
