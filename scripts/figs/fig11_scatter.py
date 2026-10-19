"""Fig. 11  Realisation scatter of the high-fidelity moduli (10 design points x 8 random arrangements) and what the two
operating modes can resolve. (a) coefficient of variation over the 8 arrangements vs the MAPE of both modes on the same
80 RVEs (models of n_H = 160, repeat 0); (b) the design point with the largest scatter: HF value of each arrangement,
hybrid prediction for each arrangement and the single parametric prediction."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, os.path.dirname(__file__))
from mfhomog import jmestyle as J
from common import FIGS, ROOT
import matplotlib.pyplot as plt

J.setup()
P = np.load(os.path.join(ROOT, 'results', 'pred_scatter_nH160_rep0.npz'))
Y = P['Y']; grp = P['idx'] // 8
names = ['$G_{12}$', '$G_{23}$', '$k_{23}$']
cv = np.array([[Y[grp == g, j].std(ddof=1) / Y[grp == g, j].mean() * 100 for j in range(3)] for g in range(10)])
e_cor = np.abs(P['BN-COR'] / Y - 1).mean(0) * 100
e_ar1 = np.abs(P['BN-AR1'] / Y - 1).mean(0) * 100
# deviation of each arrangement from its group mean = error of an ideal parametric model (noise floor)
floor = np.array([np.mean(np.abs(Y[grp == g, j] / Y[grp == g, j].mean() - 1)) * 100 for g in range(10) for j in range(3)]).reshape(10, 3).mean(0)
fig, axs = J.fig('full', 60, ncols=2, gridspec_kw=dict(width_ratios=[1.0, 1.2]))
plt.subplots_adjust(left=0.08, right=0.99, bottom=0.30, top=0.97, wspace=0.25)
ax = axs[0]; x = np.arange(3); w = 0.26
# panel (a): mean +- SD over the 10 training draws at n_H = 160 (results/scatter_eval.json, scripts/scatter_eval.py)
import json
SE = json.load(open(os.path.join(ROOT, 'results', 'scatter_eval.json')))['budgets']['160']
ek = dict(lw=0.5, capsize=1.5)
ax.bar(x - w, floor, w, color='white', edgecolor='#333333', hatch='////', lw=0.6, label='理想参数化模型（排布离散）')
ax.bar(x, SE['BN-AR1']['mean'], w, yerr=SE['BN-AR1']['sd'], error_kw=ek, color=J.C[1], edgecolor='#333333', lw=0.6, label='本文参数化模式')
ax.bar(x + w, SE['BN-COR']['mean'], w, yerr=SE['BN-COR']['sd'], error_kw=ek, color=J.C[0], edgecolor='#333333', lw=0.6, label='本文混合模式')
ax.set_xticks(x); ax.set_xticklabels(names); ax.tick_params(axis='x', which='both', top=False, bottom=False)
J.ylabel(ax, 'MAPE/%'); ax.set_ylim(0, 0.9); ax.legend(loc='upper right', fontsize=6.5)
ax.text(0.5, -0.25, '(a) 80 个 RVE 上的误差（10 次重复）', transform=ax.transAxes, ha='center', va='top')
ax = axs[1]
g = int(np.argmax(cv[:, 1])); sel = grp == g; j = 1
k = np.arange(sel.sum()) + 1
ax.plot(k, Y[sel, j], 'o', color='#111111', ms=4, mfc='none', label='高保真值')
ax.plot(k, P['BN-COR'][sel, j], 'o', color=J.C[0], ms=2.6, label='本文混合模式')
ax.plot(k, P['BN-AR1'][sel, j], 's', color=J.C[1], ms=2.6, label='本文参数化模式')
ax.plot(k, P['LF'][sel, j], '.', color=J.GREY, ms=3, label='LF ($N$=128)')
ax.set_xticks(k); J.xlabel(ax, '随机排布编号'); J.ylabel(ax, '$G_{23}/G_\\mathrm{m}$')
h, l = ax.get_legend_handles_labels(); ax.legend(h, [J.M(x)[0] for x in l], loc='best', fontsize=6.5, prop={'family': J.CJK_FONT, 'size': 6.5})
t_, f_ = J.M(f'(b) 同一设计点的 8 个排布（变异系数 {cv[g, j]:.2f}%）')
ax.text(0.5, -0.25, t_, transform=ax.transAxes, ha='center', va='top', **f_)
out = os.path.join(FIGS, 'fig11_scatter')
fig.savefig(out + '.png', dpi=600); fig.savefig(out + '.pdf')
print('saved', out, 'CV mean', cv.mean(0).round(3), 'floor', floor.round(3), 'AR1', e_ar1.round(3), 'COR', e_cor.round(3))
