"""Fig. 12  Accuracy against computing cost. (a) online CPU time per prediction vs mean MAPE at n_H = 40;
(b) offline CPU time of the training data vs mean MAPE (curves over n_H).
Solve times: clean single-core benchmark (results/cost_benchmark.json, scripts/benchmark_cost.py).
Usage: python3 scripts/figs/fig12_cost.py [test|val]"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, os.path.dirname(__file__))
from mfhomog import jmestyle as J
from common import load, STYLE, FIGS, ROOT
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else 'test'
J.setup()
d = load(src)
B = json.load(open(os.path.join(ROOT, 'results', 'cost_benchmark.json')))['summary']
t = {k: B[k]['median'] for k in ('64', '128', '256', '512', 'geometry', 'analytic', 'analytic_MT')}
tHF = t['256'] + t['512'] + t['geometry']
NB = 40
fig, axs = J.fig('full', 66, ncols=2)
plt.subplots_adjust(left=0.08, right=0.99, bottom=0.37, top=0.97, wspace=0.24)
ax = axs[0]
pts = []
for m, st in STYLE.items():
    g = d[(d.method == m) & (d.nH == NB)]
    if g.empty or m == 'MFRNP':      # MFRNP trains inside predict(); its inference time is not recorded separately
        continue
    online = g['pred_s'].median() if 'pred_s' in g else np.nan
    if m in ('BN-COR',):
        online += t['128'] + t['geometry']
    ax.plot(online, g['mape_mean3'].mean(), marker=st['mk'], color=J.C[st['c']], ls='none', ms=4, label=st['label'])
for m, cost, mk in [('LF', t['128'] + t['geometry'], 'P'), ('RICH', t['64'] + t['128'] + t['geometry'], 'X'), ('MT', t['analytic_MT'], 'h')]:
    g = d[d.method == m]
    if len(g):
        ax.plot(cost, g['mape_mean3'].mean(), marker=mk, color=J.GREY, ls='none', ms=4, label=m)
ax.axvline(tHF, color='#111111', lw=0.6, ls='--')
ax.text(tHF * 0.85, 0.97, '高保真求解', transform=ax.get_xaxis_transform(), ha='right', va='top', fontsize=7)
ax.set_xscale('log'); ax.set_yscale('log'); J.logfmt(ax, 'y')
J.xlabel(ax, '单次预测的在线计算时间/s'); J.ylabel(ax, '平均 MAPE/%')
t_, f_ = J.M(f'(a) 在线计算量($n_\\mathrm{{H}}$={NB})')
ax.text(0.5, -0.24, t_, transform=ax.transAxes, ha='center', va='top', **f_)
ax = axs[1]
for m, st in STYLE.items():
    g = d[d.method == m]
    if g.empty:
        continue
    s = g.groupby('nH')[['nL', 'mape_mean3']].mean()
    nH_ = s.index.values.astype(float); nL_ = s['nL'].values
    cLF = t['128'] + t['geometry']
    if m == 'BN-COR':        # HF samples + their coarse solve
        cost = nH_ * (tHF + t['128'])
    elif (nL_ > 0).any():    # multi-fidelity: HF samples + the extra coarse-grid samples
        cost = nH_ * tHF + (nL_ - nH_) * cLF + nH_ * t['128']
    else:
        cost = nH_ * tHF
    ax.plot(cost / 60, s['mape_mean3'], color=J.C[st['c']], ls=st['ls'], marker=st['mk'], ms=3, lw=0.9)
ax.set_xscale('log'); ax.set_yscale('log'); J.logfmt(ax, 'y'); J.logfmt(ax, 'x', (1.0, 2.0, 5.0))
J.xlabel(ax, '训练数据的离线计算时间/min'); J.ylabel(ax, '平均 MAPE/%')
ax.text(0.5, -0.24, '(b) 离线计算量', transform=ax.transAxes, ha='center', va='top')
h, l = axs[0].get_legend_handles_labels()
from matplotlib.lines import Line2D                 # MFRNP appears only in panel (b): add its legend entry
h.append(Line2D([], [], color=J.C[STYLE['MFRNP']['c']], ls=STYLE['MFRNP']['ls'], marker=STYLE['MFRNP']['mk'], ms=3, lw=0.9))
l.append('MFRNP(仅图 b)')
fig.legend(h, [J.M(x)[0] for x in l], loc='lower center', ncol=6, prop={'family': J.CJK_FONT, 'size': 7.0},
           columnspacing=0.9, handlelength=1.2, bbox_to_anchor=(0.5, 0.0))
out = os.path.join(FIGS, 'fig12_cost' + ('' if src == 'test' else '_VALLAYOUT'))
fig.savefig(out + '.png', dpi=600); fig.savefig(out + '.pdf')
print('saved', out, 'times', t, 'HF', tHF)
