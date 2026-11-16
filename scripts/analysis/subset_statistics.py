#!/usr/bin/env python3
"""Every released per-domain comparison, recomputed on a subset of the test partition.

    python3 -I subset_statistics.py <runs_dir> <repo> <subset_uids.txt|ALL> <out_dir> [family ...]

Same resampling as scripts/bootstrap_designs.py (domain-stratified resampling of source groups;
seed-pair design RNG 1234, seed-mean design RNG 1235; 10,000 replicates), restricted to the
comments listed in the subset file. Adds two-sided p-values, p = 2*min(p, 1-p), and Holm
adjustment of the two-sided values within each comparison family.
"""
import sys, json, pathlib
import numpy as np, pandas as pd
runs, repo, subset, out_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], pathlib.Path(sys.argv[4])
only = set(sys.argv[5:])
NB = 10000
out_dir.mkdir(parents=True, exist_ok=True)
keep = None if subset == 'ALL' else set(pathlib.Path(subset).read_text().split())

def macro_f1_counts(y, p):
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    den = 2 * tp + fp + fn
    return float(np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0).mean())

def holm(pvals):
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); adj, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, (m - i) * p); adj[k] = float(min(1.0, run))
    return adj

idx = pd.read_csv(repo / 'data' / 'corpus_index.csv.gz', usecols=['uid', 'video_id_hash'], dtype=str)
grp_of = dict(zip(idx['uid'], idx['video_id_hash'].where(idx['video_id_hash'].notna(), 'solo_' + idx['uid'])))
def load(run_id):
    return pd.read_csv(runs / run_id / 'preds_test.csv', usecols=['uid', 'domain', 'gold', 'pred'], dtype={'uid': str})

for sf in sorted((repo / 'results' / 'stats').glob('stats_*.json')):
    fam = sf.name.replace('stats_', '').replace('.json', '')
    if only and fam not in only:
        continue
    rel = json.load(open(sf))
    regime, baseline, seeds, scopes = rel['regime'], rel['baseline'], rel['seeds'], rel['scopes']
    results, p2_mean, p2_pair = {}, {}, {}
    for sysname in rel['comparisons']:
        results[sysname] = {}
        for scope in scopes:
            ra = [f'{regime}.{scope}.{sysname}.s{s}' for s in seeds if (runs / f'{regime}.{scope}.{sysname}.s{s}' / 'preds_test.csv').exists()]
            rb = [f'{regime}.{scope}.{baseline}.s{s}' for s in seeds if (runs / f'{regime}.{scope}.{baseline}.s{s}' / 'preds_test.csv').exists()]
            As, Bs = [load(r) for r in ra], [load(r) for r in rb]
            order = [u for u in As[0]['uid'].tolist() if keep is None or u in keep]
            def align(df):
                d = df.set_index('uid').loc[order]; return d['gold'].to_numpy(), d['pred'].to_numpy()
            ya, yb = [align(a) for a in As], [align(b) for b in Bs]
            y = ya[0][0]
            for g, _ in ya + yb: assert (g == y).all()
            pA, pB = np.stack([p for _, p in ya]), np.stack([p for _, p in yb])
            dom = As[0].set_index('uid').loc[order, 'domain'].to_numpy()
            grp = np.array([grp_of.get(u, u) for u in order])
            units = []
            for d in np.unique(dom):
                md = dom == d; gids = pd.unique(grp[md])
                units.append((gids, {g: np.flatnonzero(md & (grp == g)) for g in gids}))
            fA = np.array([macro_f1_counts(y, p) for p in pA]); fB = np.array([macro_f1_counts(y, p) for p in pB])
            e = {'n_items': int(len(y)), 'n_units': int(sum(len(g) for g, _ in units)), 'mean_A': round(float(fA.mean()), 4), 'mean_B': round(float(fB.mean()), 4),
                 'diff_mean_over_seeds': round(float(fA.mean() - fB.mean()), 4), 'n_seeds_A': len(pA), 'n_seeds_B': len(pB)}
            def resample(rng):
                take = []
                for gids, idx_by_g in units:
                    for g in rng.choice(gids, size=len(gids), replace=True): take.append(idx_by_g[g])
                return np.concatenate(take)
            rng = np.random.RandomState(1234); pairs = [(i, j) for i in range(len(pA)) for j in range(len(pB))]; diffs = np.empty(NB)
            for b in range(NB):
                i, j = pairs[rng.randint(len(pairs))]; ix = resample(rng)
                diffs[b] = macro_f1_counts(y[ix], pA[i][ix]) - macro_f1_counts(y[ix], pB[j][ix])
            lo, hi = np.percentile(diffs, [2.5, 97.5]); p1 = float((diffs <= 0).mean()); pp = min(1.0, 2 * min(p1, 1 - p1))
            e['seed_pair'] = {'ci95': [round(float(lo), 4), round(float(hi), 4)], 'p_onesided_A_gt_B': round(p1, 4), 'p_two_sided': round(pp, 4)}
            p2_pair[(sysname, scope)] = pp
            rng = np.random.RandomState(1235); kA, kB = len(pA), len(pB); diffs = np.empty(NB)
            for b in range(NB):
                ix = resample(rng); yy = y[ix]
                fa = np.array([macro_f1_counts(yy, p[ix]) for p in pA]); fb = np.array([macro_f1_counts(yy, p[ix]) for p in pB])
                diffs[b] = fa[rng.randint(kA, size=kA)].mean() - fb[rng.randint(kB, size=kB)].mean()
            lo, hi = np.percentile(diffs, [2.5, 97.5]); p1 = float((diffs <= 0).mean()); pm = min(1.0, 2 * min(p1, 1 - p1))
            e['seed_mean'] = {'ci95': [round(float(lo), 4), round(float(hi), 4)], 'p_onesided_A_gt_B': round(p1, 4), 'p_two_sided': round(pm, 4)}
            p2_mean[(sysname, scope)] = pm
            results[sysname][scope] = e
            print(fam, sysname, scope, e['diff_mean_over_seeds'], e['seed_mean']['ci95'], e['seed_pair']['ci95'], flush=True)
    for (s, d), a in holm(p2_mean).items(): results[s][d]['seed_mean']['holm_two_sided_within_family'] = round(a, 4)
    for (s, d), a in holm(p2_pair).items(): results[s][d]['seed_pair']['holm_two_sided_within_family'] = round(a, 4)
    json.dump({'regime': regime, 'baseline': baseline, 'family': fam, 'subset': subset if subset == 'ALL' else pathlib.Path(subset).name, 'n_boot': NB, 'comparisons': results},
              open(out_dir / f'stats_{fam}.json', 'w'), indent=1)
print('done')
