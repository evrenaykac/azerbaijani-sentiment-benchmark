#!/usr/bin/env python3
"""Agreement on the test partition and per batch, with bootstrap intervals.

    python3 -I agreement.py <benchmark_repo> <test_cohort.csv> <out_json> [n_boot]

Inputs are released files only: data/annotations.csv (per-annotator records of the
11,250 triple-annotated comments) and the cohort list of the test partition
(1,257 verification uids; every other triple-annotated comment is from the blind batch).
Fleiss' kappa over the three annotators (the adjudicator is excluded), percentile
bootstrap over items (10,000 replicates, seed 20261113).
"""
import sys, json, pathlib
import numpy as np, pandas as pd

repo = pathlib.Path(sys.argv[1]); cohort_csv = pathlib.Path(sys.argv[2]); out = pathlib.Path(sys.argv[3])
NB = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
LAB = {'negative': 0, 'neutral': 1, 'positive': 2}
rng = np.random.default_rng(20261113)

ann = pd.read_csv(repo / 'data' / 'annotations.csv', dtype=str).fillna('')
coh = pd.read_csv(cohort_csv, dtype=str)
ann = ann.merge(coh, on='uid', how='left')
ann['batch'] = np.where(ann['cohort'] == 'verification', 'verification', 'blind')
assert (ann[ann.gold_role == 'gold_test']['cohort'].notna()).all()
A = np.stack([ann[c].map(LAB).to_numpy() for c in ['ann1', 'ann2', 'ann3']], axis=1)
adj = ann['adjudicator'].map(LAB).to_numpy(); fin = ann['final_label'].map(LAB).to_numpy()
counts = np.stack([(A == k).sum(1) for k in range(3)], axis=1).astype(float)   # items x categories

def fleiss(c):
    n = c.sum(1)[0]; N = c.shape[0]
    Pi = ((c * c).sum(1) - n) / (n * (n - 1))
    pj = c.sum(0) / (N * n)
    Pe = (pj * pj).sum()
    return float((Pi.mean() - Pe) / (1 - Pe))

def cohen(a, b):
    po = (a == b).mean()
    pe = sum((a == k).mean() * (b == k).mean() for k in range(3))
    return float((po - pe) / (1 - pe))

def summarize(mask, name):
    c = counts[mask]; a = A[mask]; N = int(mask.sum())
    k = fleiss(c)
    idx = rng.integers(0, N, size=(NB, N))
    ks = np.array([fleiss(c[i]) for i in idx])
    lo, hi = np.percentile(ks, [2.5, 97.5])
    unan = float((c.max(1) == 3).mean()); two = float((c.max(1) == 2).mean()); none = int((c.max(1) == 1).sum())
    maj = c.argmax(1); has_maj = c.max(1) >= 2
    dep = float((adj[mask][has_maj] != maj[has_maj]).mean())
    pair = [cohen(a[:, i], a[:, j]) for i, j in [(0, 1), (0, 2), (1, 2)]]
    split = (c.max(1) == 2)
    neu_in_split = float(((c[split, 1] > 0)).mean()) if split.any() else float('nan')
    r = {'n': N, 'fleiss_kappa': round(k, 6), 'ci95': [round(float(lo), 6), round(float(hi), 6)],
         'unanimous': round(unan, 4), 'two_of_three': round(two, 4), 'no_majority_items': none,
         'adjudicator_departs_from_majority': round(dep, 4), 'pairwise_cohen': [round(x, 4) for x in pair],
         'two_of_three_splits_involving_neutral': round(neu_in_split, 4),
         'final_label_share': [round(float((fin[mask] == k_).mean()), 4) for k_ in range(3)]}
    print(f'{name:38s} n={N:5d} kappa={k:.3f} [{lo:.3f},{hi:.3f}] unanimous={unan:.3f} adj-departs={dep:.4f} pairwise={[round(x,3) for x in pair]}')
    return r

res = {}
is_test = (ann.gold_role == 'gold_test').to_numpy()
blind = (ann.batch == 'blind').to_numpy(); ver = ~blind
res['all_triple_annotated'] = summarize(np.ones(len(ann), bool), 'all triple-annotated (11,250)')
res['blind_batch_retained'] = summarize(blind, 'blind batch (retained)')
res['verification_batch'] = summarize(ver, 'verification batch')
res['test_partition'] = summarize(is_test, 'test partition')
res['test_blind'] = summarize(is_test & blind, 'test, blind cohort')
res['test_verification'] = summarize(is_test & ver, 'test, verification cohort')
for part in ['gold_train', 'gold_dev']:
    res[part + '_triple'] = summarize((ann.gold_role == part).to_numpy(), f'{part} (triple-annotated part)')
for d in ['Tech', 'Finance', 'Social', 'Retail', 'Public']:
    dm = (ann.domain == d).to_numpy()
    res[f'blind_{d}'] = summarize(blind & dm, f'blind batch, {d}')
for d in ['Tech', 'Finance', 'Social', 'Retail', 'Public']:
    dm = (ann.domain == d).to_numpy()
    res[f'test_{d}'] = summarize(is_test & dm, f'test partition, {d}')
# cohort x domain table for the test partition
tab = ann[is_test].groupby(['domain', 'batch']).size().unstack(fill_value=0)
res['test_cohort_by_domain'] = {d: {b: int(tab.loc[d, b]) for b in tab.columns} for d in tab.index}
print(tab)
# triple-annotation coverage per partition
idx = pd.read_csv(repo / 'data' / 'corpus_index.csv.gz', dtype=str).fillna('')
g = idx[idx.gold_role != '']
cov = g.groupby('gold_role').apply(lambda x: pd.Series({'n': len(x), 'triple': int((x.n_annotators == '3').sum())}), include_groups=False)
cov['share'] = (cov['triple'] / cov['n']).round(4)
print(cov); res['triple_coverage'] = {k: {'n': int(v['n']), 'triple': int(v['triple']), 'share': float(v['share'])} for k, v in cov.iterrows()}
res['triple_in_bulk'] = int(((idx.gold_role == '') & (idx.n_annotators == '3')).sum())
print('triple-annotated bulk rows:', res['triple_in_bulk'], '| verification uids:', int(ver.sum()), '| blind retained:', int(blind.sum()))
out.write_text(json.dumps(res, indent=1))
