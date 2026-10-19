"""Fig. 3  Verification of the high-fidelity data. (a) Relative deviation from the reference value
y_ref = 2 y_1024 - y_512 against grid size h = 1/N on the 12 reference RVEs (median and range), with the error of the
high-fidelity definition y_H = 2 y_512 - y_256; (b) FFT-Galerkin against the independent Q1 FE solver on the same
pixel images (20 RVEs), relative difference against N."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, os.path.dirname(__file__))
from mfhomog import jmestyle as J, design, rve, fft_solver as fs, micromech as mm, data as D
from common import FIGS, ROOT
import matplotlib.pyplot as plt

J.setup()
NAMES = ['$G_{12}$', '$G_{23}$', '$k_{23}$']
ref = D.load(os.path.join(ROOT, 'data', 'ref.jsonl'))
# coarse levels of the reference RVEs (cheap; same geometry seeds as generate_data.py: 6_000_000 + idx)
cache = os.path.join(ROOT, 'results', 'ref_coarse.json')
if os.path.exists(cache):
    coarse = json.load(open(cache))
else:
    coarse = {}
    for r_ in ref:
        p = r_['phys']; m = mm.phase(1.0, p['nu_m']); f = mm.phase(p['contrast'], p['nu_f'])
        Xc, rad = rve.generate(p['vf'], 36, p['gap'], 400, seed=r_['seed'])
        out = {}
        for N in (64, 128):
            img = rve.rasterise(Xc, rad, N)
            Ga, _, _ = fs.antiplane_shear(img, f['mu'], m['mu'], tol=1e-5)
            C, _, _ = fs.plane_strain(img, f['lam'], f['mu'], m['lam'], m['mu'], tol=1e-5)
            k, G, _ = fs.transverse_moduli(C); out[str(N)] = [float(np.trace(Ga) / 2), float(G), float(k)]
        coarse[str(r_['idx'])] = out
    json.dump(coarse, open(cache, 'w'))
lev = [64, 128, 256, 512, 1024]
err = {N: [] for N in lev}; errH = []
for r_ in ref:
    L = r_['levels']; v = lambda N: np.array([L[str(N)]['G12'], L[str(N)]['G23'], L[str(N)]['k23']]) if str(N) in L \
        else np.array(coarse[str(r_['idx'])][str(N)])
    yref = 2 * v(1024) - v(512)
    for N in lev:
        err[N].append(np.abs(v(N) / yref - 1) * 100)
    errH.append(np.abs((2 * v(512) - v(256)) / yref - 1) * 100)
fig, axs = J.fig('full', 62, ncols=2)
plt.subplots_adjust(left=0.08, right=0.99, bottom=0.30, top=0.97, wspace=0.24)
ax = axs[0]; h = 1 / np.array(lev)
for j in range(3):
    E = np.array([np.array(err[N])[:, j] for N in lev])
    ax.plot(h, np.median(E, 1), color=J.C[j], ls=J.LS[j], marker=J.MK[j], ms=3, lw=0.9, label=NAMES[j])
    ax.fill_between(h, np.percentile(E, 25, 1), np.percentile(E, 75, 1), color=J.C[j], alpha=0.15, lw=0)
    ax.plot(1 / 512 * 0.80, np.median(np.array(errH)[:, j]), marker=J.MK[j], mfc='white', mec=J.C[j], ls='none', ms=4)
hh = np.array([1 / 64, 1 / 256]); ax.plot(hh, 0.9 * hh / hh[0], color='#111111', lw=0.6)
ax.text(1 / 150, 0.33, '斜率 1', fontsize=7)
ax.text(1 / 512 * 0.80 * 1.18, 0.105, '式(8)', fontsize=7, ha='left', va='center')
ax.set_xscale('log'); ax.set_yscale('log'); J.logfmt(ax, 'y', (1.0, 3.0)); ax.minorticks_off()
ax.set_xticks(h); ax.set_xticklabels(['1/64', '1/128', '1/256', '1/512', '1/1024'])
J.xlabel(ax, '网格尺寸 $h$'); J.ylabel(ax, '相对偏差/%'); ax.legend(loc='upper left')
ax.text(0.5, -0.25, '(a) 网格收敛(空心点为式(8)的高保真值)', transform=ax.transAxes, ha='center', va='top')
ax = axs[1]
vp = os.path.join(ROOT, 'results', 'verify_solver.jsonl')
V = [json.loads(l) for l in open(vp)] if os.path.exists(vp) else []
for j in range(3):
    E = np.array([[abs(rec['levels'][str(N)]['fft'][j] / rec['levels'][str(N)]['fe'][j] - 1) * 100 for N in (64, 128, 256)] for rec in V])
    if len(E):
        ax.plot([64, 128, 256], np.median(E, 0), color=J.C[j], ls=J.LS[j], marker=J.MK[j], ms=3, lw=0.9, label=NAMES[j])
        ax.fill_between([64, 128, 256], np.percentile(E, 25, 0), np.percentile(E, 75, 0), color=J.C[j], alpha=0.15, lw=0)
ax.set_xscale('log', base=2); ax.set_yscale('log'); J.logfmt(ax, 'y', (1.0, 3.0)); ax.minorticks_off()
ax.set_xticks([64, 128, 256]); ax.set_xticklabels(['64', '128', '256'])
J.xlabel(ax, '网格分辨率 $N$'); J.ylabel(ax, 'FFT 与有限元的相对差/%'); ax.legend(loc='upper right')
ax.text(0.5, -0.25, f'(b) 与独立有限元解的比较({len(V)} 个 RVE)', transform=ax.transAxes, ha='center', va='top')
out = os.path.join(FIGS, 'fig03_verify')
fig.savefig(out + '.png', dpi=600); fig.savefig(out + '.pdf')
summ = {str(N): np.median(np.array(err[N]), 0).round(3).tolist() for N in lev}
summ['HF'] = np.median(np.array(errH), 0).round(3).tolist(); summ['HF_max'] = np.max(np.array(errH), 0).round(3).tolist()
json.dump(summ, open(os.path.join(ROOT, 'results', 'verify_summary.json'), 'w'))
print('median rel. deviation (%) per level:', summ)
