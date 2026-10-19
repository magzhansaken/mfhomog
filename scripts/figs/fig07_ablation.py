"""Fig. 7  Ablation of the hybrid model: correction without s_L input (BN-COR-X), linear correction (BN-LINCOR), no
bound normalisation (COR), no online coarse solve (BN-AR1, BN-GP, MT-GP) or no learned correction (LF). (a) mean MAPE of the three moduli, (b) mean of the maximum APE.
Usage: python3 scripts/figs/fig07_ablation.py [test|val]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, os.path.dirname(__file__))
from mfhomog import jmestyle as J
from common import load, FIGS
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else 'test'
J.setup()
d = load(src).copy()
d['max3'] = d[['max_G12', 'max_G23', 'max_k23']].mean(1)
V = [('BN-COR', '完整模型', dict(color=J.C[0])),
     ('BN-COR-X', '校正量不以粗网格解为输入', dict(color='#9ecae1')),
     ('BN-LINCOR', 'GP 校正换为线性回归', dict(color='white', hatch='xxxx')),
     ('COR', '去掉界限归一化', dict(color='white', hatch='////')),
     ('BN-AR1', '去掉在线求解（参数化模式）', dict(color=J.C[1])),
     ('BN-GP', '去掉粗网格信息', dict(color='white', hatch='....')),
     ('MT-GP', '去掉粗网格信息且只以 MT 归一化', dict(color='white', hatch='\\\\')),
     ('LF', '去掉校正（仅粗网格解）', dict(color=J.GREY))]
B = [10, 20, 40]
fig, axs = J.fig('full', 70, ncols=2)
plt.subplots_adjust(left=0.08, right=0.99, bottom=0.36, top=0.97, wspace=0.20)
w = 0.105
for k, (ax, col, lab) in enumerate(zip(axs, ['mape_mean3', 'max3'], ['平均 MAPE/%', '最大 APE 的平均值/%'])):
    for i, (m, name, st) in enumerate(V):
        g = d[d.method == m]
        if g.empty:
            continue
        mu = [g[g.nH == b][col].mean() for b in B]; sd = [g[g.nH == b][col].std() for b in B]
        x = np.arange(len(B)) + (i - 3.5) * w
        ax.bar(x, mu, w, yerr=np.nan_to_num(sd), edgecolor='#333333', lw=0.5, error_kw=dict(lw=0.5, capsize=1.2),
               label=name, **st)
    ax.set_yscale('log'); ax.set_ylim(top=ax.get_ylim()[1] * 1.6); J.logfmt(ax, 'y', (1.0, 2.0, 3.0, 5.0)); ax.set_xticks(range(len(B)))
    ax.set_xticklabels([f'$n_\\mathrm{{H}}$={b}' for b in B]); ax.tick_params(axis='x', which='both', top=False, bottom=False)
    J.ylabel(ax, lab)
    ax.text(0.5, -0.17, '(a) 平均误差' if k == 0 else '(b) 最大误差', transform=ax.transAxes, ha='center', va='top')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, columnspacing=1.0, handlelength=1.6, bbox_to_anchor=(0.5, 0.0))
out = os.path.join(FIGS, 'fig07_ablation' + ('' if src == 'test' else '_VALLAYOUT'))
fig.savefig(out + '.png', dpi=600); fig.savefig(out + '.pdf')
print('saved', out)
