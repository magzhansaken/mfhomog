"""Table 4 of the paper: online and offline cost against accuracy at n_H = 20 (test set).
Solve times: single-core benchmark medians (results/cost_benchmark.json); offline cost as in Fig. 12b
(HF sample = N 256 + 512 solves + geometry; the hybrid also needs the N = 128 solve of each HF sample;
multi-fidelity: n_L = 4 n_H coarse samples, n_H of them at the HF points).
Usage: python3 scripts/make_table4.py"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'figs'))
from common import load, ROOT

NB = 20
d = load('test'); g = d[d.nH == NB]
B = json.load(open(os.path.join(ROOT, 'results', 'cost_benchmark.json')))['summary']
t = {k: B[k]['median'] for k in ('64', '128', '256', '512', 'geometry', 'analytic_MT')}
tHF = t['256'] + t['512'] + t['geometry']; cLF = t['128'] + t['geometry']


def sci(x):
    if x >= 0.1:
        return f'{x:.2f}'
    m, e = f'{x:.1e}'.split('e'); return f'{m}×10^{int(e)}^'.replace('^-', '^−')


def mape(m):
    return f"{g[g.method == m]['mape_mean3'].mean():.2f}"


rows = [('高保真求解（*N*=256、512）', '2 次', sci(tHF), '—', '参考值'),
        ('RICH（*N*=64、128）', '2 次', sci(t['64'] + t['128'] + t['geometry']), '0', mape('RICH')),
        ('LF（*N*=128）', '1 次', sci(cLF), '0', mape('LF'))]
q = g[g.method == 'BN-COR']
rows.append(('本文混合模式', '1 次', sci(cLF + q.pred_s.median()), f'{NB * (tHF + t["128"]) / 60:.2f}', mape('BN-COR')))
q = g[g.method == 'BN-AR1']; nL = int(q.nL.iloc[0])
rows.append(('本文参数化模式', '无', sci(q.pred_s.median()), f'{(NB * tHF + (nL - NB) * cLF + NB * t["128"]) / 60:.2f}', mape('BN-AR1')))
q = g[g.method == 'BMLP']
rows.append(('BMLP', '无', sci(q.pred_s.median()), f'{NB * tHF / 60:.2f}', mape('BMLP')))
rows.append(('Mori-Tanaka', '无', sci(t['analytic_MT']), '0', mape('MT')))
L = ['| 方法 | 在线求解 | 单次预测时间/s | 训练数据 CPU 时间/min | 平均 MAPE/% |', '|---|---|---|---|---|']
L += ['| ' + ' | '.join(r) + ' |' for r in rows]
open(os.path.join(ROOT, 'paper', 'table4_test.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
