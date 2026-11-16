#!/usr/bin/env python3
"""One convention for intervals and p-values, from the released statistics.

    python3 -I two_sided_statistics.py <repo> <out_json> [<out_csv>]

Two-sided p = 2*min(p, 1-p) from the released one-sided bootstrap p; Holm within each
comparison family and jointly; number of per-domain effects whose two-sided 95% interval
excludes zero, by direction, against the number expected by chance (2.5% per direction).
"""
import sys, json, pathlib
import numpy as np
repo, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])

def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj

res = {}
ALL = []
for design, folder, suffix, ci_key, p_key in [('seed_mean', 'stats/seed_mean', '_seed_mean', 'ci95', 'p_onesided_A_gt_B'), ('seed_pair', 'stats', '', 'ci95', 'p_onesided_A_gt_B')]:
    rows = []
    for f in sorted((repo / 'results' / folder).glob(f'stats_*{suffix}.json')):
        if design == 'seed_pair' and 'seed_mean' in f.name: continue
        d = json.loads(f.read_text())
        fam = f.name.replace('stats_', '').replace('_seed_mean', '').replace('.json', '')
        for sysname, doms in d['comparisons'].items():
            for dom, v in doms.items():
                p1 = v[p_key]; lo, hi = v[ci_key]
                rows.append({'family': fam, 'system': sysname, 'domain': dom, 'baseline': d['baseline'],
                             'diff': v.get('diff_mean_over_seeds', v.get('diff_mean_over_seedpairs')), 'lo': lo, 'hi': hi, 'p1': p1, 'p2': min(1.0, 2 * min(p1, 1 - p1))})
    fams = sorted({r['family'] for r in rows})
    for fam in fams:
        idx = [i for i, r in enumerate(rows) if r['family'] == fam]
        a1 = holm(np.array([rows[i]['p1'] for i in idx])); a2 = holm(np.array([rows[i]['p2'] for i in idx]))
        for j, i in enumerate(idx): rows[i]['holm1_family'] = float(a1[j]); rows[i]['holm2_family'] = float(a2[j])
    j1 = holm(np.array([r['p1'] for r in rows])); j2 = holm(np.array([r['p2'] for r in rows]))
    ALL.extend(dict(r, design=design) for r in rows)
    pos = [r for r in rows if r['lo'] > 0]; neg = [r for r in rows if r['hi'] < 0]
    res[design] = {'n_comparisons': len(rows), 'families': {f: sum(1 for r in rows if r['family'] == f) for f in fams},
                   'interval_excludes_zero_positive': len(pos), 'interval_excludes_zero_negative': len(neg),
                   'expected_by_chance_per_direction': round(0.025 * len(rows), 2),
                   'min_holm_one_sided_family': round(min(r['holm1_family'] for r in rows), 4), 'min_holm_two_sided_family': round(min(r['holm2_family'] for r in rows), 4),
                   'min_holm_one_sided_joint': round(float(j1.min()), 4), 'min_holm_two_sided_joint': round(float(j2.min()), 4),
                   'positive': [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in sorted(pos, key=lambda r: r['p2'])],
                   'negative': [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in sorted(neg, key=lambda r: r['p2'])]}
    print(f'== {design}: {len(rows)} comparisons in {len(fams)} families {res[design]["families"]}')
    print(f'   intervals excluding zero: {len(pos)} positive, {len(neg)} negative; expected by chance per direction: {0.025 * len(rows):.2f}')
    print(f'   smallest Holm-adjusted p within family: one-sided {res[design]["min_holm_one_sided_family"]}, two-sided {res[design]["min_holm_two_sided_family"]}; joint: one-sided {res[design]["min_holm_one_sided_joint"]}, two-sided {res[design]["min_holm_two_sided_joint"]}')
    for r in sorted(pos, key=lambda r: r['p2']):
        print(f'   + {r["family"]:28s} {r["system"]:16s} {r["domain"]:8s} {r["diff"]:+.4f} [{r["lo"]:+.4f},{r["hi"]:+.4f}] p1={r["p1"]:.4f} p2={r["p2"]:.4f} Holm(two-sided, family)={r["holm2_family"]:.3f} Holm(one-sided)={r["holm1_family"]:.3f}')
    for r in sorted(neg, key=lambda r: r['p2']):
        print(f'   - {r["family"]:28s} {r["system"]:16s} {r["domain"]:8s} {r["diff"]:+.4f} [{r["lo"]:+.4f},{r["hi"]:+.4f}] p2={r["p2"]:.4f} Holm(two-sided, family)={r["holm2_family"]:.3f}')
out.write_text(json.dumps(res, indent=1))
if len(sys.argv) > 3:
    import csv
    cols = ['design', 'family', 'baseline', 'system', 'domain', 'diff', 'lo', 'hi', 'p1', 'p2', 'holm2_family']
    with open(sys.argv[3], 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['design', 'family', 'baseline', 'system', 'domain', 'difference', 'ci95_low', 'ci95_high', 'p_one_sided', 'p_two_sided', 'holm_two_sided_within_family'])
        for r in ALL: w.writerow([(f'{r[c]:.4f}' if isinstance(r[c], float) else r[c]) for c in cols])
