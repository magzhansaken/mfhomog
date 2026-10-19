"""Fig. 8  Predicted against high-fidelity values on the 300 test RVEs (n_H = 20, repeat 0): proposed hybrid,
strongest published baseline at this budget and the raw coarse-grid solve.
Usage: python3 scripts/figs/fig08_parity.py BASELINE   (BASELINE read from paper/numbers_test.json if omitted)"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, os.path.dirname(__file__))
from mfhomog import jmestyle as J
from common import STYLE, FIGS, ROOT
import matplotlib.pyplot as plt

J.setup()
P = np.load(os.path.join(ROOT, 'results', 'pred_test_nH20_rep0.npz'))
base = sys.argv[1] if len(sys.argv) > 1 else json.load(open(os.path.join(ROOT, 'paper', 'numbers_test.json')))['strongest_published']['20']['method']
Y = P['Y']; names = ['$G_{12}/G_\\mathrm{m}$', '$G_{23}/G_\\mathrm{m}$', '$k_{23}/k_\\mathrm{m}$']
fig, axs = J.fig('full', 62, ncols=3)
plt.subplots_adjust(left=0.07, right=0.99, bottom=0.33, top=0.97, wspace=0.30)
series = [('LF', P['LF'], dict(color=J.GREY, marker='.', ms=2.5, label='LF ($N$=128)')),
          (base, P[base] if base in P else P['RICH'], dict(color=(J.C[STYLE[base]['c']] if base in STYLE else '#555555'), marker='^', ms=2.5,
                                                          mfc='none', mew=0.5, label=STYLE.get(base, {'label': base})['label'])),
          ('BN-COR', P['BN-COR'], dict(color=J.C[0], marker='o', ms=2.2, label='本文混合模式'))]
for j, ax in enumerate(axs):
    lo, hi = Y[:, j].min() * 0.95, Y[:, j].max() * 1.05
    ax.plot([lo, hi], [lo, hi], color='#111111', lw=0.6)
    for name, Yp, st in series:
        e = np.mean(np.abs(Yp[:, j] / Y[:, j] - 1)) * 100
        ax.plot(Y[:, j], Yp[:, j], ls='none', **{**st, 'label': f"{st['label']} ({e:.2f}%)"})
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    J.xlabel(ax, '高保真值 ' + names[j]); J.ylabel(ax, '预测值 ' + names[j])
    leg = ax.legend(*[(h, [J.M(x)[0] for x in l]) for h, l in [ax.get_legend_handles_labels()]][0], loc='lower right',
                    prop={'family': J.CJK_FONT, 'size': 6.3}, handletextpad=0.2, borderaxespad=0.2, markerscale=1.5)
    t_, f_ = J.M(f'({"abc"[j]}) {names[j]}')
    ax.text(0.5, -0.27, t_, transform=ax.transAxes, ha='center', va='top', **f_)
out = os.path.join(FIGS, 'fig08_parity')
fig.savefig(out + '.png', dpi=600); fig.savefig(out + '.pdf')
print('saved', out, 'baseline', base)
