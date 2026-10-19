"""CPU time (process_time, robust to other jobs on the machine) of every fidelity level, single process, one core.
Usage: python3 scripts/benchmark_cost.py OUT.json            (all levels)
       python3 scripts/benchmark_cost.py OUT.json --analytic-only   (adds 'analytic_MT' = Mori-Tanaka / HS lower
       bound alone, mean of 1000 calls per design point, to an existing OUT.json)"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mfhomog import design, rve, fft_solver as fs, micromech as mm

X = design.sobol(16, 909)
if '--analytic-only' in sys.argv:
    J = json.load(open(sys.argv[1])); J['raw']['analytic_MT'] = []
    for x in X:
        p = design.to_phys(x); m = mm.phase(1.0, p['nu_m']); f = mm.phase(p['contrast'], p['nu_f'])
        t = time.process_time()
        for _ in range(1000):
            mm.mori_tanaka(p['vf'], f, m)
        J['raw']['analytic_MT'].append((time.process_time() - t) / 1000)
    v = J['raw']['analytic_MT']
    J['summary']['analytic_MT'] = dict(mean=float(np.mean(v)), std=float(np.std(v)), median=float(np.median(v)))
    json.dump(J, open(sys.argv[1], 'w'), indent=1); print(J['summary']['analytic_MT']); sys.exit(0)
out = {str(N): [] for N in (64, 128, 256, 512)}
out['geometry'] = []; out['analytic'] = []
for i, x in enumerate(X):
    p = design.to_phys(x); m = mm.phase(1.0, p['nu_m']); f = mm.phase(p['contrast'], p['nu_f'])
    t = time.process_time(); Xc, r = rve.generate(p['vf'], 36, p['gap'], 400, seed=9_000_000 + i); out['geometry'].append(time.process_time() - t)
    t = time.process_time(); mm.hs_bounds(p['vf'], f, m); mm.halpin_tsai_set(p['vf'], f, m); out['analytic'].append(time.process_time() - t)
    for N in (64, 128, 256, 512):
        img = rve.rasterise(Xc, r, N)
        t = time.process_time()
        fs.antiplane_shear(img, f['mu'], m['mu'], tol=1e-5); fs.plane_strain(img, f['lam'], f['mu'], m['lam'], m['mu'], tol=1e-5)
        out[str(N)].append(time.process_time() - t)
summary = {k: dict(mean=float(np.mean(v)), std=float(np.std(v)), median=float(np.median(v))) for k, v in out.items()}
summary['HF_256_plus_512'] = dict(mean=float(np.mean(np.array(out['256']) + np.array(out['512']))))
json.dump(dict(raw=out, summary=summary), open(sys.argv[1], 'w'), indent=1)
print(json.dumps(summary, indent=1))
