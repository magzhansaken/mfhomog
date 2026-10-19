"""Fig. 10  Empirical coverage and mean relative width of the 95 % prediction intervals (mean of the three moduli).
Usage: python3 scripts/figs/fig10_uq.py [test|val]"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, os.path.dirname(__file__))
from mfhomog import jmestyle as J
from common import load, STYLE, FIGS
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else 'test'
J.setup()
d = load(src).copy()
d['cov3'] = d[['cov95_G12', 'cov95_G23', 'cov95_k23']].mean(1)
d['wid3'] = d[['width95_G12', 'width95_G23', 'width95_k23']].mean(1)
fig, axs = J.fig('full', 66, ncols=2)
plt.subplots_adjust(left=0.08, right=0.99, bottom=0.36, top=0.97, wspace=0.22)
for k, (ax, col, lab) in enumerate(zip(axs, ['cov3', 'wid3'], ['95% 区间覆盖率/%', '区间相对宽度/%'])):
    for m, st in STYLE.items():
        g = d[(d.method == m) & d[col].notna()]
        if g.empty:
            continue
        s = g.groupby('nH')[col].agg(['mean', 'std'])
        ax.errorbar(s.index, s['mean'], yerr=s['std'], color=J.C[st['c']], ls=st['ls'], marker=st['mk'], ms=3.0, lw=0.9,
                    elinewidth=0.5, capsize=1.5, label=st['label'])
    cf = os.path.join(os.path.dirname(FIGS), '..', 'results', 'conformal.json')
    if src == 'test' and os.path.exists(cf):   # post-hoc split-conformal calibration on the validation set (section 4.5)
        C = json.load(open(cf)); nb = sorted(int(k) for k in C)
        yv = [C[str(b)]['cov_conformal_mean3' if col == 'cov3' else 'width_conformal_mean3'] for b in nb]
        ax.plot(nb, yv, color=J.C[0], ls='--', marker='o', mfc='white', ms=3.2, lw=0.9, label='本文混合模式（共形校准）')
    ax.set_xscale('log'); ax.set_xticks([10, 20, 40, 80, 160]); ax.set_xticklabels(['10', '20', '40', '80', '160'])
    ax.minorticks_off()
    if col == 'cov3':
        ax.axhline(95, color='#111111', lw=0.6, ls='--'); ax.set_ylim(-2, 108); ax.text(10.5, 97, '名义值 95%', fontsize=7, va='bottom')
    else:
        ax.set_yscale('log'); J.logfmt(ax, 'y'); ax.set_ylim(0.08, 25)
    J.xlabel(ax, '高保真样本数 $n_\\mathrm{H}$'); J.ylabel(ax, lab)
    ax.text(0.5, -0.24, '(a) 覆盖率' if k == 0 else '(b) 区间宽度', transform=ax.transAxes, ha='center', va='top')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, columnspacing=1.0, handlelength=2.0, bbox_to_anchor=(0.5, 0.0))
out = os.path.join(FIGS, 'fig10_uq' + ('' if src == 'test' else '_VALLAYOUT'))
fig.savefig(out + '.png', dpi=600); fig.savefig(out + '.pdf')
print('saved', out)
