#!/usr/bin/env python3
"""Label composition of the pre-audit training pools, by held-out target.

    python3 -I teacher_label_shares.py <earlier_split_runs_dir> <out_json> [<label_control_runs_dir>]

Reads the pool composition recorded in metrics.json of the plain XLM-R runs with
teacher-labeled comments (lodo.<target>.xlmr_ft.s42) and with human labels only
(lodo.<target>.xlmr_ft.s42.frac000). The teacher-labeled part is their difference.
Shares are negative/neutral/positive.

With the third argument, the same record is read for the control runs on the audited
split (Public Services target and pooled): S is the pool with added teacher-labeled
comments, H the human-labeled pool, T the human-labeled pool relabeled by the teacher.
"""
import sys, json, pathlib
runs, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
ctrl = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else None
LABS = ['negative', 'neutral', 'positive']
def counts(run, base=None):
    comp = json.loads(((base or runs) / run / 'metrics.json').read_text())['composition']
    c = [0, 0, 0]
    for key, n in comp['train_by_domain_label'].items(): c[LABS.index(key.split('|')[1])] += n
    assert sum(c) == comp['train_total'], run
    return c
share = lambda c: [round(x / sum(c), 3) for x in c]
res = {}
for target in ['Tech', 'Finance', 'Social', 'Retail', 'Public']:
    pool, human = counts(f'lodo.{target}.xlmr_ft.s42'), counts(f'lodo.{target}.xlmr_ft.s42.frac000')
    teacher = [a - b for a, b in zip(pool, human)]
    res[target] = {'pool': sum(pool), 'pool_share': share(pool), 'human': sum(human), 'human_share': share(human), 'teacher': sum(teacher), 'teacher_share': share(teacher)}
    print(target, res[target])
if ctrl is not None:
    for name, scope in (('audited_Public', 'lodo.Public'), ('audited_all', 'pooled.all')):
        s, h, tt = (counts(f'{scope}.xlmr_ft.s42.{k}', ctrl) for k in ('silver', 'human', 'relabel'))
        teacher = [a - b for a, b in zip(s, h)]
        res[name] = {'S_share': share(s), 'H_share': share(h), 'T_share': share(tt), 'teacher': sum(teacher), 'teacher_share': share(teacher)}
        print(name, res[name])
out.write_text(json.dumps(res, indent=1))
