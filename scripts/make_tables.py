"""Table 3 of the paper and paper/numbers.json (every number quoted in the text is read from here).
Usage: python3 scripts/make_tables.py [test|val]"""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'figs'))
from mfhomog.stats import holm, paired
from common import load, ROOT

src = sys.argv[1] if len(sys.argv) > 1 else 'test'
d = load(src)
OURS, PARAM = 'BN-COR', 'BN-AR1'
ORDER = ['MT', 'LF', 'RICH', 'GP', 'BN-GP', 'MLP', 'BMLP', 'VRN', 'XGB', 'AR1', 'NARGP', 'BN-NARGP', 'MFNN', 'TLNN', 'MFRNP',
         'COR', 'BN-COR-X', 'BN-LINCOR', 'MT-GP', 'BN-AR1', 'BN-COR']
BUD = [10, 20, 40, 80, 160]
metrics = ['mape_mean3', 'mape_G12', 'mape_G23', 'mape_k23']   # derived constants: results/test_derived_v2.csv (PROTOCOL 16:04)
num = {}

# summary per method and budget
summ = d.groupby(['method', 'nH'])[metrics + ['max_G12', 'max_G23', 'max_k23', 'cov95_G12', 'cov95_G23', 'cov95_k23',
                                             'width95_G12', 'width95_G23', 'width95_k23', 'viol_G12', 'viol_G23',
                                             'viol_k23', 'fit_s']].agg(['mean', 'std', 'count'])
num['summary'] = {f'{m}|{n}': {f'{c}_{s}': float(v) for (c, s), v in row.items()} for (m, n), row in summ.iterrows()}

# paired tests of OURS against every other method, Holm within (budget, metric)
tests = []
for nH, g in d.groupby('nH'):
    a = g[g.method == OURS].set_index('rep')
    for met in metrics:
        block = []
        for b in sorted(set(g.method) - {OURS}):
            bb = g[g.method == b].set_index('rep'); reps = a.index.intersection(bb.index)
            if len(reps) < 5:
                continue
            p, w, red, rb = paired(a.loc[reps, met].values, bb.loc[reps, met].values)
            block.append(dict(nH=int(nH), metric=met, baseline=b, n=int(len(reps)), p=float(p), wins=int(w),
                              med_rel_red=float(red), rank_biserial=float(rb)))
        for t, pa in zip(block, holm([t['p'] for t in block])):
            t['p_holm'] = float(pa); tests.append(t)
T = pd.DataFrame(tests)
T.to_csv(os.path.join(ROOT, 'results', f'{src}_tests_vs_{OURS}.csv'), index=False)
num['tests'] = tests

# strongest competitor per budget (lowest mean mape_mean3 among all methods except ours and its own ablations)
excl = {OURS, 'COR', 'BN-COR-X', 'BN-LINCOR', 'MT-GP'}   # ours and its ablations
best = {}
for nH in sorted(d.nH.unique()):
    s = d[(d.nH == nH) & ~d.method.isin(excl)].groupby('method')['mape_mean3'].mean().sort_values()
    best[int(nH)] = dict(method=s.index[0], mape=float(s.iloc[0]), ranking=[(k, float(v)) for k, v in s.items()])
num['strongest_competitor'] = best
# strongest published baseline (our own variants BN-* and COR excluded)
bestpub = {}
for nH in sorted(d.nH.unique()):
    s_ = d[(d.nH == nH) & ~d.method.str.startswith('BN') & ~d.method.isin(['COR', 'LINCOR', 'MT-GP'])].groupby('method')['mape_mean3'].mean().sort_values()
    bestpub[int(nH)] = dict(method=s_.index[0], mape=float(s_.iloc[0]))
num['strongest_published'] = bestpub

# Table 3 (markdown, three-line in the docx builder): mean +- std of mape_mean3 and Holm-adjusted p against OURS
def cell(m, nH):
    g = d[(d.method == m) & (d.nH == nH)]['mape_mean3']
    if g.empty:
        return '—'
    s = f'{g.mean():.2f}±{g.std():.2f}' if g.std() > 1e-9 else f'{g.mean():.2f}'
    if m != OURS:
        t = T[(T.nH == nH) & (T.metric == 'mape_mean3') & (T.baseline == m)]
        if len(t):
            s += '*' if t.p_holm.iloc[0] < 0.05 and t.med_rel_red.iloc[0] > 0 else ''
    return s


lines = ['| 方法 | ' + ' | '.join(f'*n*~H~={b}' for b in BUD) + ' |', '|---' * (len(BUD) + 1) + '|']
for m in ORDER:
    if m in set(d.method):
        lines.append(f'| {m} | ' + ' | '.join(cell(m, b) for b in BUD) + ' |')
open(os.path.join(ROOT, 'paper', f'table3_{src}.md'), 'w').write('\n'.join(lines) + '\n')
json.dump(num, open(os.path.join(ROOT, 'paper', f'numbers_{src}.json'), 'w'), indent=1)
print('\n'.join(lines))
for nH, b in best.items():
    o = d[(d.method == OURS) & (d.nH == nH)]['mape_mean3'].mean()
    bp = bestpub[nH]
    print(nH, 'strongest competitor', b['method'], round(b['mape'], 3), '| strongest published', bp['method'], round(bp['mape'], 3),
          '| ours', round(o, 3), '| reduction vs competitor', round(100 * (1 - o / b['mape']), 1), '% vs published', round(100 * (1 - o / bp['mape']), 1), '%')
