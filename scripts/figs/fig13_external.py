"""Fig. 13  External data sets. (a) Short natural-fibre composites (3D FE, independent 60-row test set): mean MAPE of
E3, E12, G13, G12 against the number of training geometry groups (x 7 contrasts); dashed line = the dataset authors'
GP trained on all 2520 rows. (b) Mechanical MNIST: MAPE of the strain energy on the 10000 test examples; published CNN
and transfer-learning CNN errors computed from the authors' prediction files (scripts/mnist_published.py).
The hybrid model on this set is COR (log coordinate; the strain energy has no bounds), drawn as COR in Fig. 7."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, os.path.dirname(__file__))
from mfhomog import jmestyle as J
from common import FIGS, ROOT
import matplotlib.pyplot as plt

J.setup()
fig, axs = J.fig('full', 64, ncols=2, gridspec_kw=dict(width_ratios=[1.15, 1.0]))
plt.subplots_adjust(left=0.08, right=0.99, bottom=0.30, top=0.97, wspace=0.28)
ax = axs[0]
d = pd.read_csv(os.path.join(ROOT, 'results', 'ext_nf_final.csv'))
pub = d[d.method == 'PUB-GP-2520']['mape_mean'].iloc[0]
NF = [('BN-GP', 'BN-GP', '#111111', '-', 'o'), ('GP', 'GP', J.C[2], '-.', '^'),
      ('HT-GP', 'HT-GP', '#9a9a9a', (0, (5, 1, 1, 1)), 'D'), ('MLP', 'MLP', '#555555', ':', 'v')]
for m, lab, c, ls, mk in NF:
    g = d[d.method == m]
    if g.empty:
        continue
    q = g.groupby('n_groups')['mape_mean'].quantile([0.25, 0.5, 0.75]).unstack()   # median and IQR (robust to the
    ax.errorbar(q.index * 7, q[0.5], yerr=[q[0.5] - q[0.25], q[0.75] - q[0.5]], color=c, ls=ls, marker=mk,  # divergent GP repeat)
                ms=3, lw=0.9, elinewidth=0.5, capsize=1.5, label=lab)
ht = d[d.method == 'HT']['mape_mean'].mean()
ax.axhline(ht, color=J.GREY, ls='--', lw=0.7, label='Halpin-Tsai')
ax.axhline(pub, color='#111111', ls=(0, (6, 2)), lw=0.8, label='原作者 GP(2520 个样本)')
ax.set_xscale('log'); ax.set_yscale('log'); J.logfmt(ax, 'y'); ax.minorticks_off(); ax.set_ylim(0.9, 40)
ax.set_xticks([42, 84, 168, 336, 672]); ax.set_xticklabels(['42', '84', '168', '336', '672'])
J.xlabel(ax, '训练样本数'); J.ylabel(ax, 'MAPE 中位数/%'); ax.legend(loc='upper right', fontsize=6.3, ncol=2, columnspacing=0.8)
ax.text(0.5, -0.25, '(a) 天然纤维短纤维复合材料', transform=ax.transAxes, ha='center', va='top')
ax = axs[1]
m = pd.read_csv(os.path.join(ROOT, 'results', 'ext_mnist.csv'))
pubm = json.load(open(os.path.join(ROOT, 'results', 'mnist_published.json')))
rows = [('CNN', pubm['CNN_nH100_mape'], np.nan, 'white', None), ('TL-CNN', pubm['TL_CNN_nH100_LF1000_mape'], np.nan, '#dddddd', None)]
for meth, lab, col, hat in [('LF + log-linear map', '标定低保真解', J.GREY, None), ('GP (HF only)', 'GP', J.C[2], None),
                            ('Corrector GP (LF at query)', 'COR', 'white', '////')]:
    g = m[(m.method == meth) & (m.nH == 100)]
    rows.append((lab, g.mape.mean(), g.mape.std(), col, hat))
labels = [r[0] for r in rows]; vals = [r[1] for r in rows]
x = np.arange(len(vals))
for xi, (lab, v, sd, col, hat) in zip(x, rows):     # error bars (SD over 10 repeats) only where repeats exist
    ax.bar(xi, v, 0.6, color=col, edgecolor='#333333', lw=0.6, hatch=hat)
    if np.isfinite(sd):
        ax.errorbar(xi, v, yerr=sd, color='#111111', lw=0.5, capsize=1.5)
for xi, v in zip(x, vals):
    ax.text(xi, v * 1.04 + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5); ax.tick_params(axis='x', which='both', top=False, bottom=False)
ax.set_ylim(0, 9.2); J.ylabel(ax, 'MAPE/%')
t_, f_ = J.M('(b) Mechanical MNIST($n_\\mathrm{H}$=100)')
ax.text(0.5, -0.25, t_, transform=ax.transAxes, ha='center', va='top', **f_)
out = os.path.join(FIGS, 'fig13_external')
fig.savefig(out + '.png', dpi=600); fig.savefig(out + '.pdf')
print('saved', out, 'NF published GP', pub)
