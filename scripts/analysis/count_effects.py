#!/usr/bin/env python3
"""Count per-domain effects by level of support in a directory of stats_*.json files.

    python3 -I count_effects.py <stats_dir> [<out_json>]
"""
import sys, json, glob, os
DOMS = ['Tech', 'Finance', 'Social', 'Retail', 'Public']
def load(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, 'stats_*.json'))):
        s = json.load(open(f)); comps = s['comparisons']
        it = comps.items() if isinstance(comps, dict) else [(c.get('system', '?'), c) for c in comps]
        for sysname, c in it:
            for dom in DOMS:
                if dom not in c: continue
                e = c[dom]
                rows.append(dict(fam=s.get('family', os.path.basename(f)), base=s.get('baseline'), sys=sysname, dom=dom,
                                 diff=e['diff_mean_over_seeds'],
                                 plo=e['seed_pair']['ci95'][0], phi=e['seed_pair']['ci95'][1], ph=e['seed_pair']['holm_two_sided_within_family'],
                                 mlo=e['seed_mean']['ci95'][0], mhi=e['seed_mean']['ci95'][1], mh=e['seed_mean']['holm_two_sided_within_family']))
    return rows
def summary(rows):
    pos = lambda lo, hi: lo > 0
    neg = lambda lo, hi: hi < 0
    o = {'n': len(rows)}
    for tag, lo, hi, h in (('seed_pair', 'plo', 'phi', 'ph'), ('seed_mean', 'mlo', 'mhi', 'mh')):
        P = [r for r in rows if r[lo] > 0]; N = [r for r in rows if r[hi] < 0]
        o[tag] = dict(positive=len(P), negative=len(N),
                      positive_after_holm=[f"{r['fam']}:{r['sys']}:{r['dom']}" for r in P if r[h] < 0.05],
                      negative_after_holm=[f"{r['fam']}:{r['sys']}:{r['dom']} {r['diff']:+.3f} [{r[lo]:+.3f},{r[hi]:+.3f}] p_adj={r[h]:.3g}" for r in N if r[h] < 0.05],
                      positive_vs_xlmr=[f"{r['sys']}:{r['dom']}" for r in P if r['fam'].endswith('vs_xlmr')],
                      min_holm=min(r[h] for r in rows))
    return o
if __name__ == '__main__':
    rows = load(sys.argv[1]); s = summary(rows)
    print(json.dumps(s, indent=1))
    if len(sys.argv) > 2: json.dump(dict(summary=s, rows=rows), open(sys.argv[2], 'w'), indent=1)
