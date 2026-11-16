"""Shared helpers for the round-2 checks (stored predictions only)."""
import json, pathlib
import numpy as np, pandas as pd

LAB = ['negative', 'neutral', 'positive']
DOMS = ['Tech', 'Finance', 'Social', 'Retail', 'Public']
SEEDS = [13, 21, 42, 87, 100]

def macro_f1(y, p):
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3).astype(float)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    den = 2 * tp + fp + fn
    return float(np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0).mean())

def per_class_f1(y, p):
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3).astype(float)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)

def group_map(repo):
    idx = pd.read_csv(pathlib.Path(repo) / 'data' / 'corpus_index.csv.gz', usecols=['uid', 'video_id_hash'], dtype=str)
    has = idx['video_id_hash'].notna() & (idx['video_id_hash'] != '')
    grp = idx['video_id_hash'].where(has, 'solo_' + idx['uid'])
    return dict(zip(idx['uid'], grp)), dict(zip(idx['uid'], has))

def load_runs(runs_dir, regime, scope, system, seeds=SEEDS, tag=''):
    out = []
    for s in seeds:
        f = pathlib.Path(runs_dir) / f'{regime}.{scope}.{system}.s{s}{tag}' / 'preds_test.csv'
        if f.exists():
            out.append(pd.read_csv(f, usecols=['uid', 'domain', 'gold', 'pred'], dtype={'uid': str}))
    return out

def align(dfs, uid_order):
    return np.stack([d.set_index('uid').loc[uid_order, 'pred'].to_numpy() for d in dfs])

def make_units(dom, grp):
    units = []
    for d in np.unique(dom):
        md = dom == d
        gids = pd.unique(grp[md])
        units.append((gids, {g: np.flatnonzero(md & (grp == g)) for g in gids}))
    return units

def resample(units, rng):
    take = []
    for gids, idx_by_g in units:
        pick = rng.choice(gids, size=len(gids), replace=True)
        take.extend(idx_by_g[g] for g in pick)
    return np.concatenate(take)

def both_designs(y, pA, pB, units, n_boot=10000, seed_pair=1234, seed_mean=1235):
    """Seed-pair (pre-specified) and seed-mean (two-level) bootstrap of macro-F1(A) - macro-F1(B)."""
    fA = np.array([macro_f1(y, p) for p in pA]); fB = np.array([macro_f1(y, p) for p in pB])
    obs = float(fA.mean() - fB.mean())
    rng = np.random.RandomState(seed_pair)
    pairs = [(i, j) for i in range(len(pA)) for j in range(len(pB))]
    dp = np.empty(n_boot)
    for b in range(n_boot):
        i, j = pairs[rng.randint(len(pairs))]
        ix = resample(units, rng)
        dp[b] = macro_f1(y[ix], pA[i][ix]) - macro_f1(y[ix], pB[j][ix])
    rng = np.random.RandomState(seed_mean)
    kA, kB = len(pA), len(pB); dm = np.empty(n_boot)
    for b in range(n_boot):
        ix = resample(units, rng); yy = y[ix]
        fa = np.array([macro_f1(yy, p[ix]) for p in pA]); fb = np.array([macro_f1(yy, p[ix]) for p in pB])
        dm[b] = fa[rng.randint(kA, size=kA)].mean() - fb[rng.randint(kB, size=kB)].mean()
    def summ(d):
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {'ci95': [round(float(lo), 4), round(float(hi), 4)], 'p_onesided_A_gt_B': round(float((d <= 0).mean()), 4)}
    return {'diff_mean_over_seeds': round(obs, 4), 'kA': kA, 'kB': kB, 'seed_pair': summ(dp), 'seed_mean': summ(dm)}
