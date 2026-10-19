"""Clean single-core CPU time (process_time) of the FFT-Galerkin solver (tol 1e-5) and the periodic Q1 FE solver
(sparse direct LU) on the same pixel images: the first 8 verification RVEs of scripts/verify_solver.py
(design seed 808, microstructure seeds 8_000_000 + i), N = 128 and 256. Both moduli problems (anti-plane + plane strain).
Usage: OPENBLAS_NUM_THREADS=1 python3 scripts/benchmark_fe.py results/fe_benchmark.json"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mfhomog import design, rve, fft_solver as fs, fe_check as fe, micromech as mm

X = design.sobol(20, 808)
out = {str(N): dict(fft=[], fe=[]) for N in (128, 256)}
for i, x in enumerate(X[:8]):
    p = design.to_phys(x); m = mm.phase(1.0, p['nu_m']); f = mm.phase(p['contrast'], p['nu_f'])
    Xc, r = rve.generate(p['vf'], 36, p['gap'], 400, seed=8_000_000 + i)
    for N in (128, 256):
        img = rve.rasterise(Xc, r, N)
        t = time.process_time()
        fs.antiplane_shear(img, f['mu'], m['mu'], tol=1e-5); fs.plane_strain(img, f['lam'], f['mu'], m['lam'], m['mu'], tol=1e-5)
        out[str(N)]['fft'].append(time.process_time() - t)
        t = time.process_time()
        fe.antiplane_fe(img, f['mu'], m['mu']); fe.plane_strain_fe(img, f['lam'], f['mu'], m['lam'], m['mu'])
        out[str(N)]['fe'].append(time.process_time() - t)
    print(i, {N: (round(v['fft'][-1], 2), round(v['fe'][-1], 2)) for N, v in out.items()}, flush=True)
summ = {N: dict(fft_median=float(np.median(v['fft'])), fe_median=float(np.median(v['fe'])),
                ratio_median=float(np.median(np.array(v['fe']) / np.array(v['fft'])))) for N, v in out.items()}
json.dump(dict(raw=out, summary=summ), open(sys.argv[1], 'w'), indent=1); print(json.dumps(summ, indent=1))
