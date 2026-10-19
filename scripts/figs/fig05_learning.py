"""Fig. 5  Test error against the number of high-fidelity samples (mean +- std over repeats).
Usage: python3 scripts/figs/fig05_learning.py [test|val]   (val only to check the layout)"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, os.path.dirname(__file__))
from mfhomog import jmestyle as J
from common import load, STYLE, REF, FIGS
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullLocator

src = sys.argv[1] if len(sys.argv) > 1 else 'test'
J.setup()
d = load(src)
panels = [('mape_G12', '$G_{12}$'), ('mape_G23', '$G_{23}$'), ('mape_k23', '$k_{23}$'), ('mape_mean3', '三个模量平均')]
fig, axs = J.fig('full', 112, ncols=2, nrows=2)
plt.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.98, wspace=0.22, hspace=0.48)
for k, (ax, (col, name)) in enumerate(zip(axs.ravel(), panels)):
    for mname, st in STYLE.items():
        g = d[d.method == mname]
        if g.empty:
            continue
        s = g.groupby('nH')[col].agg(['mean', 'std'])
        ax.errorbar(s.index, s['mean'], yerr=s['std'], color=J.C[st['c']], ls=st['ls'], marker=st['mk'], ms=3.2,
                    lw=1.0 if mname.startswith('BN') else 0.8, elinewidth=0.5, capsize=1.5, label=st['label'],
                    zorder=3 if mname.startswith('BN') else 2)
    for mname, st in REF.items():
        g = d[d.method == mname]
        if g.empty:
            continue
        v = g[col].mean()
        ax.axhline(v, color=J.GREY, ls=st['ls'], lw=0.8, label=st['label'], zorder=1)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.margins(y=0.08)
    ax.set_xticks([10, 20, 40, 80, 160]); ax.set_xticklabels(['10', '20', '40', '80', '160'])
    ax.minorticks_off()
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); ax.yaxis.set_minor_locator(NullLocator())
    J.xlabel(ax, '高保真样本数 $n_\\mathrm{H}$'); J.ylabel(ax, 'MAPE/%')
    t, f = J.M(f'({"abcd"[k]}) {name}')
    ax.text(0.5, -0.29, t, transform=ax.transAxes, ha='center', va='top', fontsize=7.5, **f)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, [J.M(x)[0] for x in l], loc='lower center', ncol=6, prop={'family': J.CJK_FONT, 'size': 7.0},
           columnspacing=1.0, handlelength=2.2, bbox_to_anchor=(0.5, 0.0))
out = os.path.join(FIGS, 'fig05_learning' + ('' if src == 'test' else '_VALLAYOUT'))
fig.savefig(out + '.png', dpi=600); fig.savefig(out + '.pdf')
print('saved', out)
