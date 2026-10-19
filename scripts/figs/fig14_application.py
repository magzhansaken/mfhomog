"""Fig. 14  Application example (section 4.9): target transverse modulus E2 = 3.0 Em for a glass/epoxy-like system.
(a) E2/Em against V_f predicted by the parametric mode (n_H = 40, repeat 1 draw) with its 95 % interval, the
Mori-Tanaka estimate (= HS lower bound of k23, G23, G12 mapped through the same engineering relations), the target
and the selected V_f; (b) the 8 new random arrangements at the selected design: high-fidelity value and hybrid-mode
prediction per arrangement, parametric prediction with its 95 % interval.
Inputs: results/application_curve.csv (scripts/application_curve.py), results/application.json (scripts/application.py)."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, os.path.dirname(__file__))
from mfhomog import jmestyle as J
from common import FIGS, ROOT
import matplotlib.pyplot as plt

J.setup()
c = pd.read_csv(os.path.join(ROOT, 'results', 'application_curve.csv'))
A = json.load(open(os.path.join(ROOT, 'results', 'application.json')))
vf, T = A['vf_star'], A['target']
fig, axs = J.fig('full', 60, ncols=2, gridspec_kw=dict(width_ratios=[1.0, 1.0]))
plt.subplots_adjust(left=0.075, right=0.99, bottom=0.27, top=0.97, wspace=0.27)
PAR, HYB = J.C[1], J.C[0]

ax = axs[0]
ax.fill_between(c.vf, c.E2_lo95, c.E2_up95, color=PAR, alpha=0.25, lw=0, label='95% 预测区间')
ax.plot(c.vf, c.E2, color=PAR, ls='-', lw=1.0, label='本文参数化模式')
ax.plot(c.vf, c.E2_HSlow, color=J.GREY, ls='--', lw=0.8, label='MT')
ax.axhline(T, color='#111111', ls=':', lw=0.6, label='目标值'); ax.axvline(vf, color='#111111', ls=':', lw=0.6)
ax.plot(np.full(len(A['hf_E2']), vf), A['hf_E2'], 'o', ms=3.0, mfc='#111111', mec='white', mew=0.4, label='高保真值')
J.text(ax, vf + 0.006, 1.62, f'$V_\\mathrm{{f}}$={vf:.3f}', fontsize=7.0)
ax.set_xlim(0.20, 0.65); ax.set_ylim(1.4, 4.4)
J.xlabel(ax, '纤维体积分数 $V_\\mathrm{f}$'); J.ylabel(ax, '$E_2/E_\\mathrm{m}$')
J.legend(ax, loc='upper left', fontsize=7.0, handlelength=1.8, ncol=2, columnspacing=1.0)
ax.text(0.5, -0.27, '(a) 横向模量随纤维体积分数的变化', transform=ax.transAxes, ha='center', va='top')

ax = axs[1]
lo, up = A['E2_param_95']; k = np.arange(1, len(A['hf_E2']) + 1)
ax.axhspan(lo, up, color=PAR, alpha=0.25, lw=0)
ax.axhline(A['E2_param'], color=PAR, ls='-', lw=1.0, label='本文参数化模式')
ax.axhline(T, color='#111111', ls=':', lw=0.6, label='目标值')
ax.plot(k, A['hf_E2'], 'o', ms=3.5, mfc='#111111', mec='#111111', label='高保真值')
ax.plot(k, A['hybrid_E2'], 'o', ms=5.0, mfc='none', mec=HYB, mew=0.9, label='本文混合模式')
ax.set_xlim(0.4, 8.6); ax.set_xticks(k); ax.set_ylim(2.94, 3.10)
J.xlabel(ax, '随机排布编号'); J.ylabel(ax, '$E_2/E_\\mathrm{m}$')
J.legend(ax, loc='upper left', fontsize=7.0, ncol=2, columnspacing=1.0, handlelength=1.8)
t_, f_ = J.M(f'(b) $V_\\mathrm{{f}}$={vf:.3f} 处的 8 个随机排布')
ax.text(0.5, -0.27, t_, transform=ax.transAxes, ha='center', va='top', **f_)
out = os.path.join(FIGS, 'fig14_application')
fig.savefig(out + '.png', dpi=600); fig.savefig(out + '.pdf')
print('saved', out)
