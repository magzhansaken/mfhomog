"""Fig. 9  Absolute percentage error of G23 over the design space (V_f, minimum gap g) on the 300 test RVEs, n_H = 40,
repeat 0: raw coarse solve, Richardson extrapolation, strongest published baseline and the proposed hybrid.
Usage: python3 scripts/figs/fig09_errmap.py [BASELINE]"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, os.path.dirname(__file__))
from mfhomog import jmestyle as J
from common import STYLE, FIGS, ROOT
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

J.setup()
P = np.load(os.path.join(ROOT, 'results', 'pred_test_nH40_rep0.npz'))
base = sys.argv[1] if len(sys.argv) > 1 else json.load(open(os.path.join(ROOT, 'paper', 'numbers_test.json')))['strongest_published']['40']['method']
X, Y = P['X'], P['Y']; j = 1
seq = LinearSegmentedColormap.from_list('seq', ['#deebf7', '#6baed6', '#2a78d6', '#08306b'])
cols = [('LF', P['LF'], 'LF ($N$=128)'), ('RICH', P['RICH'], 'RICH'),
        (base, P[base], STYLE.get(base, {'label': base})['label']), ('BN-COR', P['BN-COR'], '本文混合模式')]
fig, axs = J.fig('full', 52, ncols=4, sharey=True)
plt.subplots_adjust(left=0.07, right=0.90, bottom=0.33, top=0.97, wspace=0.10)
norm = LogNorm(0.01, 20)
for k, (ax, (name, Yp, lab)) in enumerate(zip(axs, cols)):
    e = np.abs(Yp[:, j] / Y[:, j] - 1) * 100
    o = np.argsort(e)
    sc = ax.scatter(X[o, 0], P['gap'][o], c=np.clip(e[o], 0.01, 20), cmap=seq, norm=norm, s=6, lw=0)
    ax.set_yscale('log'); J.logfmt(ax, 'y', (1.0, 2.0, 5.0))
    J.xlabel(ax, '$V_\\mathrm{f}$')
    if k == 0:
        J.ylabel(ax, '最小间隙系数 $g$')
    t_, f_ = J.M(f'({"abcd"[k]}) {lab}, 平均 {e.mean():.2f}%')
    ax.text(0.5, -0.27, t_, transform=ax.transAxes, ha='center', va='top', **f_)
cax = fig.add_axes([0.915, 0.33, 0.012, 0.64]); cb = fig.colorbar(sc, cax=cax)
cb.set_label('$G_{23}$ 的 APE/%', fontfamily=J.CJK_FONT) if False else cb.set_label('APE/%'); cb.ax.tick_params(direction='in', width=0.5, length=2)
from matplotlib.ticker import FuncFormatter, FixedLocator
cb.ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10])); cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); cb.ax.minorticks_off()
out = os.path.join(FIGS, 'fig09_errmap')
fig.savefig(out + '.png', dpi=600); fig.savefig(out + '.pdf')
print('saved', out)
