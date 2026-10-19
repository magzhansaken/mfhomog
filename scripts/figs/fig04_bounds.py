"""Fig. 4  Bound-normalised coordinate on the training pool (400 samples; no test data used).
(a) G23/Gm against V_f; (b) relative position t in the HS interval; (c) correction s_H - s_L (N = 128) against the gap."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from mfhomog import jmestyle as J, data as D, surrogates as S
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

J.setup()
ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
pool = D.load(os.path.join(ROOT, 'data', 'pool.jsonl'))
X, Y, LO, UP = D.arrays(pool, 'HF'); _, YL, _, _ = D.arrays(pool, 128)
vf = X[:, 0]; lc = X[:, 1]; gap = np.array([r['phys']['gap'] for r in pool])
t = (Y - LO) / (UP - LO)
d = S.to_s(Y, LO, UP, 'logit') - S.to_s(YL, LO, UP, 'logit')
cmap = LinearSegmentedColormap.from_list('seq', ['#c6dbef', '#2a78d6', '#08306b'])
j = 1  # G23
fig, ax = J.fig('full', 52, ncols=3)
plt.subplots_adjust(left=0.07, right=0.90, bottom=0.30, top=0.97, wspace=0.42)
kw = dict(c=lc, cmap=cmap, s=5, lw=0, vmin=np.log10(5), vmax=2)
J.xlabel(ax[0], '纤维体积分数 $V_\\mathrm{f}$'); J.ylabel(ax[0], '$G_{23}/G_\\mathrm{m}$')
# (b) 10-fold CV error of a model linear in x, in the log, the Mori-Tanaka-normalised log(y/L) and the bound-normalised
#     coordinate (all 3 targets)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
Xs = (X - X.mean(0)) / X.std(0); cv = {}
for kind in ('log', 'mt', 'logit'):
    Sv = S.to_s(Y, LO, UP, kind); e = []
    for k in range(3):
        pr = cross_val_predict(LinearRegression(), Xs, Sv[:, k], cv=10)
        e.append(np.mean(np.abs(S.from_s(pr, LO[:, k], UP[:, k], kind) / Y[:, k] - 1)) * 100)
    cv[kind] = e
pos = np.arange(3); w = 0.27
b1 = ax[1].bar(pos - w, cv['log'], w, color='white', edgecolor='#333333', lw=0.6, hatch='////', label='对数坐标')
b3 = ax[1].bar(pos, cv['mt'], w, color='#bdbdbd', edgecolor='#333333', lw=0.6, label='MT 归一化坐标')
b2 = ax[1].bar(pos + w, cv['logit'], w, color=J.C[0], edgecolor='#333333', lw=0.6, label='界限归一化坐标')
for bb in (b1, b3, b2):
    for r in bb:
        ax[1].text(r.get_x() + r.get_width() / 2, r.get_height() + 0.08, f'{r.get_height():.2f}', ha='center', va='bottom',
                   fontsize=6.5, rotation=90)
ax[1].set_xticks(pos); ax[1].set_xticklabels(['$G_{12}$', '$G_{23}$', '$k_{23}$']); ax[1].set_ylim(0, 10)
ax[1].tick_params(axis='x', which='both', top=False)
J.ylabel(ax[1], '交叉验证 MAPE/%'); ax[1].legend(loc='upper right', handlelength=1.2, borderaxespad=0.2)
sc = ax[2].scatter(gap, d[:, j], **kw); ax[0].scatter(vf, Y[:, j], **kw)
ax[2].axhline(0, color=J.GREY, lw=0.5)
J.xlabel(ax[2], '最小间隙系数 $g$'); J.ylabel(ax[2], '校正量 $s_\\mathrm{H}-s_\\mathrm{L}$')
cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
cb.set_label('$\\lg(E_\\mathrm{f}/E_\\mathrm{m})$'); cb.ax.tick_params(direction='in', width=0.6, length=2)
for a, l in zip(ax, ['(a) 物理坐标下的 $G_{23}$', '(b) 线性模型误差', '(c) 粗网格校正量']):
    t_, f_ = J.M(l); a.text(0.5, -0.33, t_, transform=a.transAxes, ha='center', va='top', fontsize=7.5, **f_)
out = os.path.join(ROOT, 'paper', 'figs', 'fig04_bounds')
fig.savefig(out + '.png', dpi=600); fig.savefig(out + '.pdf')
json.dump({k: [float(v) for v in vs] for k, vs in cv.items()}, open(os.path.join(ROOT, 'results', 'fig04_linear_cv.json'), 'w'), indent=1)
print('linear CV MAPE', cv); print('G23 range', Y[:, j].min(), Y[:, j].max(), 't range', t[:, j].min(), t[:, j].max(),
      'corr range', d[:, j].min(), d[:, j].max())
