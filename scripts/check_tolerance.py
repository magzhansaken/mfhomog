"""CG tolerance check: effective moduli at tol 1e-5 vs 1e-7 on 4 RVEs (N = 256; design seed 919, seeds 9_100_000+i)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mfhomog import design, rve, fft_solver as fs, micromech as mm
res = []
for i, x in enumerate(design.sobol(4, 919)):
    p = design.to_phys(x); m = mm.phase(1.0, p['nu_m']); f = mm.phase(p['contrast'], p['nu_f'])
    Xc, r = rve.generate(p['vf'], 36, p['gap'], 400, seed=9_100_000 + i); img = rve.rasterise(Xc, r, 256)
    out = []
    for tol in (1e-5, 1e-7):
        Ga, _, _ = fs.antiplane_shear(img, f['mu'], m['mu'], tol=tol)
        C, _, _ = fs.plane_strain(img, f['lam'], f['mu'], m['lam'], m['mu'], tol=tol, maxiter=6000)
        k, G, _ = fs.transverse_moduli(C); out.append(np.array([np.trace(Ga) / 2, G, k]))
    res.append(float(np.abs(out[0] / out[1] - 1).max()))
json.dump(dict(max_rel_diff=res), open('results/tolerance_check.json', 'w'), indent=1); print(res)
