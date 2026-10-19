"""Fig. 6  Distribution over repeats of the mean MAPE of the three moduli, all methods, n_H = 10, 20, 40.
Methods sorted by median at n_H = 20; '*' = Holm-adjusted Wilcoxon p < 0.05 against the proposed hybrid.
Usage: python3 scripts/figs/fig06_boxes.py [test|val]"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, os.path.dirname(__file__))
from mfhomog import jmestyle as J
from common import load, FIGS, ROOT
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullLocator

src = sys.argv[1] if len(sys.argv) > 1 else 'test'
J.setup()
d = load(src)
tp = os.path.join(ROOT, 'results', f'{src}_tests_vs_BN-COR.csv')
T = pd.read_csv(tp) if os.path.exists(tp) else pd.DataFrame(columns=['nH', 'metric', 'baseline', 'p_holm', 'med_rel_red'])
det = d.groupby('method')['mape_mean3'].std().fillna(0)
order = d[d.nH == 20].groupby('method')['mape_mean3'].median().sort_values(ascending=False).index.tolist()
fig, axs = J.fig('full', 78, ncols=3, sharey=True, sharex=True)
plt.subplots_adjust(left=0.11, right=0.99, bottom=0.20, top=0.98, wspace=0.08)
for k, (ax, nH) in enumerate(zip(axs, [10, 20, 40])):
    data = [d[(d.nH == nH) & (d.method == m)]['mape_mean3'].values for m in order]
    bp = ax.boxplot(data, vert=False, widths=0.6, showfliers=True, patch_artist=True,
                    flierprops=dict(marker='o', ms=1.8, mfc='none', mec='#555555', mew=0.4),
                    medianprops=dict(color='#111111', lw=0.9), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                    boxprops=dict(lw=0.6))
    for patch, m in zip(bp['boxes'], order):
        patch.set_facecolor(J.C[0] if m == 'BN-COR' else (J.C[1] if m == 'BN-AR1' else 'white'))
    for i, m in enumerate(order):
        t = T[(T.nH == nH) & (T.metric == 'mape_mean3') & (T.baseline == m)]
        if len(t) and t.p_holm.iloc[0] < 0.05 and t.med_rel_red.iloc[0] > 0:
            ax.text(1.0, i + 1, '*', transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=8)
    ax.set_xscale('log'); ax.minorticks_off()
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis='y', which='both', right=False, left=False)
    J.xlabel(ax, '三个模量的平均 MAPE/%')
    t_, f_ = J.M(f'({"abc"[k]}) $n_\\mathrm{{H}}$={nH}')
    ax.text(0.5, -0.20, t_, transform=ax.transAxes, ha='center', va='top', fontsize=7.5, **f_)
axs[0].set_yticks(range(1, len(order) + 1)); axs[0].set_yticklabels(order)
out = os.path.join(FIGS, 'fig06_boxes' + ('' if src == 'test' else '_VALLAYOUT'))
fig.savefig(out + '.png', dpi=600); fig.savefig(out + '.pdf')
print('saved', out)
