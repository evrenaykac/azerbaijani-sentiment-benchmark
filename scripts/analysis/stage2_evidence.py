#!/usr/bin/env python3
"""Direct evidence on the Stage-2 baseline, from stored predictions only.

    python3 -I stage2_evidence.py <earlier_split_runs_dir> <out_json>

For every LODO target, for plain XLM-R fine-tuning and the full recipe, with the
teacher-labeled pool (Stage 2, default tag) and with human labels only (Stage 2b,
tag frac000): per-seed macro-F1, per-class F1, predicted class shares, gold class
shares, training-pool label shares, dev score of the selected checkpoint, steps run.
Also a post-hoc prior correction on the stored logits: logit_c - log(train prior_c).
"""
import sys, json, pathlib, collections
import numpy as np, pandas as pd

runs = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2])
LAB = ['negative', 'neutral', 'positive']
DOMS = ['Tech', 'Finance', 'Social', 'Retail', 'Public']
SEEDS = [13, 42, 100]

def f1s(y, p):
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3).astype(float)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    den = 2 * tp + fp + fn
    f = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    return f, float(f.mean()), cm

res = {}
for dom in DOMS:
    for system in ['xlmr_ft', 'full']:
        for tag, stage in [('', 'stage2_teacher'), ('.frac000', 'stage2b_human')]:
            rows = []
            for s in SEEDS:
                d = runs / f'lodo.{dom}.{system}.s{s}{tag}'
                if not (d / 'preds_test.csv').exists():
                    continue
                df = pd.read_csv(d / 'preds_test.csv')
                m = json.loads((d / 'metrics.json').read_text())
                y = df['gold'].to_numpy(); p = df['pred'].to_numpy()
                f, mf, cm = f1s(y, p)
                assert abs(mf - m['test']['macro_f1']) < 1e-9, (d, mf, m['test']['macro_f1'])
                comp = m['composition']['train_by_domain_label']
                pool = collections.Counter()
                for k, v in comp.items():
                    pool[k.split('|')[1]] += v
                tot = sum(pool.values()); prior = np.array([pool[l] / tot for l in LAB])
                logits = df[['logit_neg', 'logit_neu', 'logit_pos']].to_numpy()
                assert (logits.argmax(1) == p).mean() > 0.999
                p_adj = (logits - np.log(prior)).argmax(1)
                fa, mfa, _ = f1s(y, p_adj)
                rows.append({
                    'seed': s, 'macro_f1': round(mf, 4), 'f1': [round(float(x), 3) for x in f],
                    'pred_share': [round(float(x), 3) for x in np.bincount(p, minlength=3) / len(p)],
                    'gold_share': [round(float(x), 3) for x in np.bincount(y, minlength=3) / len(y)],
                    'recall': [round(float(cm[i, i] / cm[i].sum()), 3) for i in range(3)],
                    'train_prior': [round(float(x), 3) for x in prior], 'train_total': tot,
                    'bulk_silver': m['composition'].get('bulk_silver', 0),
                    'dev_best_macro_f1': round(m['dev_best_macro_f1'], 4), 'steps_run': m['hyperparams']['steps_run'],
                    'best_epoch': m['best_epoch'], 'n_test': int(len(y)),
                    'macro_f1_prior_corrected': round(mfa, 4),
                    'pred_share_prior_corrected': [round(float(x), 3) for x in np.bincount(p_adj, minlength=3) / len(p)],
                })
            if rows:
                res[f'{dom}.{system}.{stage}'] = {
                    'mean_macro_f1': round(float(np.mean([r['macro_f1'] for r in rows])), 4),
                    'sd_macro_f1': round(float(np.std([r['macro_f1'] for r in rows])), 4),
                    'mean_macro_f1_prior_corrected': round(float(np.mean([r['macro_f1_prior_corrected'] for r in rows])), 4),
                    'mean_pred_share': [round(float(np.mean([r['pred_share'][i] for r in rows])), 3) for i in range(3)],
                    'seeds': rows}
out.write_text(json.dumps(res, indent=1))
for dom in DOMS:
    print(f'== {dom}  gold share (neg/neu/pos): {res[f"{dom}.xlmr_ft.stage2_teacher"]["seeds"][0]["gold_share"]}  n={res[f"{dom}.xlmr_ft.stage2_teacher"]["seeds"][0]["n_test"]}')
    for system in ['xlmr_ft', 'full']:
        for stage in ['stage2_teacher', 'stage2b_human']:
            r = res.get(f'{dom}.{system}.{stage}')
            if not r: continue
            s0 = r['seeds'][0]
            print(f'  {system:8s} {stage:15s} mean {r["mean_macro_f1"]:.3f} (sd {r["sd_macro_f1"]:.3f}) seeds {[x["macro_f1"] for x in r["seeds"]]} '
                  f'pred share {r["mean_pred_share"]} train prior {s0["train_prior"]} pool {s0["train_total"]} dev {[x["dev_best_macro_f1"] for x in r["seeds"]]} '
                  f'| prior-corrected mean {r["mean_macro_f1_prior_corrected"]:.3f}')
