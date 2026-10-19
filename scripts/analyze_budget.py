"""Aggregate budget results: mean +- std over repeats, paired Wilcoxon tests (ours vs each baseline) with Holm
correction per (budget, metric), wins and median relative reduction.
Usage: python3 scripts/analyze_budget.py RESULTS.csv OURS OUT_PREFIX"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mfhomog.stats import holm, paired

res, ours, outp = sys.argv[1], sys.argv[2], sys.argv[3]
r = pd.read_csv(res)
metrics = ['mape_G12', 'mape_G23', 'mape_k23', 'mape_E2', 'mape_nu23', 'mape_mean3']
summ = r.groupby(['nH', 'method'])[metrics + ['fit_s', 'data_cpu_s']].agg(['mean', 'std'])
summ.to_csv(outp + '_summary.csv')
rows = []
for nH, g in r.groupby('nH'):
    a = g[g.method == ours].set_index('rep')
    for m in metrics:
        tests = []
        for b in sorted(set(g.method) - {ours}):
            bb = g[g.method == b].set_index('rep')
            reps = a.index.intersection(bb.index)
            if len(reps) < 5:
                continue
            p, w, red, rb = paired(a.loc[reps, m].values, bb.loc[reps, m].values)
            tests.append(dict(nH=nH, metric=m, baseline=b, n=len(reps), p=p, wins=w, med_rel_red=red, rank_biserial=rb,
                              ours_mean=a.loc[reps, m].mean(), ours_std=a.loc[reps, m].std(),
                              base_mean=bb.loc[reps, m].mean(), base_std=bb.loc[reps, m].std()))
        adj = holm([t['p'] for t in tests]) if tests else []
        for t, pa in zip(tests, adj):
            t['p_holm'] = pa; rows.append(t)
pd.DataFrame(rows).to_csv(outp + '_tests.csv', index=False)
print(summ.xs('mean', axis=1, level=1)[['mape_G12', 'mape_G23', 'mape_k23', 'mape_E2', 'mape_mean3']].round(3).to_string())
