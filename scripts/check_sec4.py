"""Every number quoted in section 4.1 of the paper, recomputed from the result files (test set).
Usage: python3 scripts/check_sec4.py  -> prints and writes results/numbers_sec41.json"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'figs'))
from common import load, ROOT

d = load('test'); B = [10, 20, 40, 80, 160]
OURS = 'BN-COR'
COMP = ['MT', 'LF', 'RICH', 'GP', 'MLP', 'BMLP', 'VRN', 'XGB', 'AR1', 'NARGP', 'MFNN', 'TLNN', 'MFRNP', 'BN-GP', 'BN-NARGP', 'BN-AR1']
ABL = ['COR', 'BN-COR-X', 'BN-LINCOR', 'MT-GP']
m = lambda meth, col: d[d.method == meth].groupby('nH')[col].mean()
out = {}
out['ours_mean3'] = m(OURS, 'mape_mean3').round(3).tolist()
# lowest among the 16 comparison methods, overall and per modulus
for col in ['mape_mean3', 'mape_G12', 'mape_G23', 'mape_k23']:
    tab = {k: m(k, col) for k in COMP + [OURS] + ABL}
    out[f'ours_lowest_vs_comparisons_{col}'] = [bool(all(tab[OURS][b] < tab[k][b] for k in COMP if b in tab[k])) for b in B]
    out[f'methods_below_ours_{col}'] = {b: [k for k in COMP + ABL if b in tab[k] and tab[k][b] < tab[OURS][b]] for b in B}
# reductions against own parametric variants (all budgets)
red = []
for k in ['BN-GP', 'BN-AR1', 'BN-NARGP']:
    red += list(1 - m(OURS, 'mape_mean3') / m(k, 'mape_mean3'))
out['red_vs_BN_variants_min_max_%'] = [round(100 * min(red), 1), round(100 * max(red), 1)]
r2 = d[d.method == OURS].groupby('nH')[['r2_G12', 'r2_G23', 'r2_k23']].mean()
out['ours_r2_min'] = round(float(r2.values.min()), 4)
out['LF'] = round(float(m('LF', 'mape_mean3').mean()), 3); out['RICH'] = round(float(m('RICH', 'mape_mean3').mean()), 3)
out['ours10_over_RICH_%'] = round(100 * m(OURS, 'mape_mean3')[10] / m('RICH', 'mape_mean3')[10], 1)
out['MFRNP'] = m('MFRNP', 'mape_mean3').round(2).tolist()
out['MT_per_modulus'] = [round(float(m('MT', c).mean()), 2) for c in ['mape_G12', 'mape_G23', 'mape_k23']]
# derived constants: corrected values after the nu12 sign fix (results/test_derived_v2.csv, PROTOCOL 16:04)
import pandas as _pd
dv = _pd.read_csv(os.path.join(ROOT, 'results', 'test_derived_v2.csv'))
mv = lambda meth, col: dv[dv.method == meth].groupby('nH')[col].mean()
for k in [OURS, 'BMLP']:
    out[f'{k}_E2'] = mv(k, 'mape_E2').round(3).tolist()
out['ours_nu23'] = mv(OURS, 'mape_nu23').round(3).tolist()
out['ours_E1_max'] = round(float(mv(OURS, 'mape_E1').max()), 4); out['ours_nu12_max'] = round(float(mv(OURS, 'mape_nu12').max()), 3)
viol = {k: d[(d.method == k) & (d.nH == 10)][['viol_G12', 'viol_G23', 'viol_k23']].mean().mean() for k in COMP + ABL + [OURS]}
out['viol_nH10_%'] = {k: round(float(v), 1) for k, v in viol.items()}
out['param_BN-AR1'] = m('BN-AR1', 'mape_mean3').round(3).tolist()
json.dump(out, open(os.path.join(ROOT, 'results', 'numbers_sec41.json'), 'w'), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
