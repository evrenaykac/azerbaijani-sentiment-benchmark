#!/usr/bin/env python3
"""Test-sampling uncertainty of the encoder-versus-LLM comparisons.

    python3 -I encoder_vs_llm.py <runs_dir> <repo> <out_json> [n_boot]

The language-model figures are single runs, so their seed variance is unknown. What the stored
predictions do allow is a paired cluster bootstrap over the test comments of the encoder's seed
mean minus the single LLM run (encoder seeds resampled, LLM run fixed).
"""
import sys, json, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import *
runs, repo, out = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
NB = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
grp_of, _ = group_map(repo)
def llm(path):
    return pd.read_csv(repo / 'results' / path, usecols=['uid', 'domain', 'gold', 'pred'], dtype={'uid': str})
JOBS = [('pooled', 'all', 'mdeberta_ft', 'qlora_llama31', 'llm/llama31/preds_test.csv'), ('pooled', 'all', 'mdeberta_ft', 'qlora_qwen25', 'llm/qwen25/preds_test.csv'),
        ('pooled', 'all', 'mdeberta_ft', 'qlora_qwen3', 'llm/qwen3/preds_test.csv'), ('pooled', 'all', 'mdeberta_ft', 'gpt4o_3shot', 'llm/openai/preds_gpt4o_3shot.csv'),
        ('pooled', 'all', 'xlmr_ft', 'qlora_llama31', 'llm/llama31/preds_test.csv'), ('pooled', 'all', 'xlmr_ft', 'gpt4o_3shot', 'llm/openai/preds_gpt4o_3shot.csv'),
        ('lodo', 'Public', 'mdeberta_ft', 'qlora_qwen25_lodo_Public', 'llm/qwen25_lodo_Public/preds_test.csv'), ('lodo', 'Tech', 'mdeberta_ft', 'qlora_qwen25_lodo_Tech', 'llm/qwen25_lodo_Tech/preds_test.csv'),
        ('lodo', 'Public', 'xlmr_ft', 'qlora_qwen25_lodo_Public', 'llm/qwen25_lodo_Public/preds_test.csv'), ('lodo', 'Tech', 'xlmr_ft', 'qlora_qwen25_lodo_Tech', 'llm/qwen25_lodo_Tech/preds_test.csv')]
res = {}
for regime, scope, enc, name, path in JOBS:
    dA = load_runs(runs, regime, scope, enc); dB = [llm(path)]
    base = dA[0]; uids = base['uid'].tolist()
    assert set(uids) == set(dB[0]['uid']), (name, len(uids), len(dB[0]))
    y = base['gold'].to_numpy(); dom = base['domain'].to_numpy(); grp = np.array([grp_of[u] for u in uids])
    assert (dB[0].set_index('uid').loc[uids, 'gold'].to_numpy() == y).all()
    r = both_designs(y, align(dA, uids), align(dB, uids), make_units(dom, grp), n_boot=NB)
    r['encoder_mean'] = round(float(np.mean([macro_f1(y, p) for p in align(dA, uids)])), 4); r['llm_single_run'] = round(macro_f1(y, align(dB, uids)[0]), 4)
    res[f'{regime}.{scope}.{enc}_vs_{name}'] = r
    print(f'{regime:7s}{scope:7s} {enc:12s} {r["encoder_mean"]:.3f} vs {name:26s} {r["llm_single_run"]:.3f}  diff {r["diff_mean_over_seeds"]:+.4f}  seed-mean CI {r["seed_mean"]["ci95"]}  seed-pair CI {r["seed_pair"]["ci95"]}', flush=True)
out.write_text(json.dumps(res, indent=1))
