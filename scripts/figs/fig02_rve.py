"""Fig. 2  Random RVEs and FFT fields. (a)(b) arrangements at V_f = 0.30 and 0.62; (c) the same fibre pair on the
128 and 512 grids (staircase error at a narrow gap); (d) longitudinal shear stress field, N = 512."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, os.path.dirname(__file__))
from mfhomog import jmestyle as J, rve, fft_solver as fs, micromech as mm
from common import FIGS
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle

J.setup()
seq = LinearSegmentedColormap.from_list('seq', ['#f7fbff', '#9ecae1', '#2a78d6', '#08306b'])
fig = plt.figure(figsize=(J.FULL_MM * J.MM, 50 * J.MM))
W = 0.205; xs = [0.02, 0.255, 0.49, 0.725]
axs = [fig.add_axes([x, 0.20, W, W * 171 / 50]) for x in xs]
geo = [(0.30, 0.25, 101), (0.62, 0.03, 202)]
for ax, (vf, g, seed) in zip(axs[:2], geo):
    X, r = rve.generate(vf, 36, g, 400, seed=seed)
    img = rve.rasterise(X, r, 512)
    ax.imshow(img.T, origin='lower', cmap='Greys', vmin=-0.3, vmax=1.25, extent=(0, 1, 0, 1), interpolation='nearest')
# (c) narrowest gap of the dense RVE on two grids
X, r = rve.generate(0.62, 36, 0.03, 400, seed=202)
D = X[:, None, :] - X[None, :, :]; D -= np.round(D); dist = np.sqrt((D ** 2).sum(-1)) + np.eye(len(X))
mid = (X[None, :, :] + D / 2) % 1.0                       # midpoints of all pairs (minimum image)
inside = np.all((mid > 0.15) & (mid < 0.85), -1)
dd = np.where(inside, dist, np.inf); i, j = np.unravel_index(np.argmin(dd), dd.shape)
c = mid[i, j]
half = 2.2 * r
ax = axs[2]
fine = rve.rasterise(X, r, 512).astype(float)
ax.imshow(fine.T, origin='lower', cmap='Greys', vmin=-0.3, vmax=1.6, extent=(0, 1, 0, 1), interpolation='nearest')
coarse = rve.rasterise(X, r, 128).astype(float)
up = np.kron(coarse, np.ones((8, 8)))                  # 128 grid shown on the 1024 sub-grid (nearest)
g = (np.arange(1024) + 0.5) / 1024
ax.contour(g, g, up.T, levels=[0.5], colors='#111111', linewidths=0.6)
for k in range(len(X)):
    for sx_ in (-1, 0, 1):
        for sy_ in (-1, 0, 1):
            ax.add_patch(Circle(X[k] + [sx_, sy_], r, fill=False, ec=J.C[1], lw=0.8))
ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half)
# (d) longitudinal shear stress, dense RVE, contrast 50, load along x
f, m = mm.phase(50.0, 0.22), mm.phase(1.0, 0.35)
img = rve.rasterise(X, r, 512)
_, _, _, flds = fs.antiplane_shear(img, f['mu'], m['mu'], tol=1e-6, return_fields=True)
sx = flds[0][0] / m['mu']
im = axs[3].imshow(sx.T, origin='lower', cmap=seq, extent=(0, 1, 0, 1), interpolation='nearest',
                   vmin=0, vmax=np.quantile(sx, 0.995))
axs[1].add_patch(Rectangle(c - half, 2 * half, 2 * half, fill=False, ec=J.C[1], lw=0.9))
for ax in axs:
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_linewidth(0.5)
cax = fig.add_axes([xs[3] + W + 0.006, 0.20, 0.010, W * 171 / 50])
cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(direction='in', width=0.5, length=2, labelsize=6.5)
labels = ['(a) $V_\\mathrm{f}$=0.30', '(b) $V_\\mathrm{f}$=0.62', '(c) 窄缝处的 $N$=128 离散', '(d) $\\sigma_{13}/(G_\\mathrm{m}\\bar{\\gamma})$']
for ax, l in zip(axs, labels):
    t_, f_ = J.M(l); ax.text(0.5, -0.06, t_, transform=ax.transAxes, ha='center', va='top', fontsize=7.5, **f_)
out = os.path.join(FIGS, 'fig02_rve')
fig.savefig(out + '.png', dpi=600); fig.savefig(out + '.pdf')
print('saved', out, 'closest gap / r =', (dist[i, j] - 2 * r) / r)
