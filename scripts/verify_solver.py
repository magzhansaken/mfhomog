"""Independent verification of the FFT-Galerkin solver against a periodic Q1 finite-element solver on the same pixel
images (20 design points not used anywhere else; design seed 808, microstructure seeds 8_000_000 + i).
For each point and resolution N in {64, 128, 256}: G12, G23, k23 from FFT (tol 1e-5) and FE (direct solve).
Usage: python3 scripts/verify_solver.py OUT.jsonl"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mfhomog import design, rve, fft_solver as fs, fe_check as fe, micromech as mm

outp = sys.argv[1]
done = set()
if os.path.exists(outp):
    done = {json.loads(l)['idx'] for l in open(outp)}
X = design.sobol(20, 808)
for i, x in enumerate(X):
    if i in done:
        continue
    p = design.to_phys(x)
    m = mm.phase(1.0, p['nu_m']); f = mm.phase(p['contrast'], p['nu_f'])
    Xc, r = rve.generate(p['vf'], 36, p['gap'], 400, seed=8_000_000 + i)
    rec = dict(idx=i, x=[float(v) for v in x], phys=p, levels={})
    for N in (64, 128, 256):
        img = rve.rasterise(Xc, r, N)
        t = time.time()
        Ga, _, _ = fs.antiplane_shear(img, f['mu'], m['mu'], tol=1e-5)
        C, _, _ = fs.plane_strain(img, f['lam'], f['mu'], m['lam'], m['mu'], tol=1e-5)
        k, G, _ = fs.transverse_moduli(C); t_fft = time.time() - t
        t = time.time()
        Gfe = fe.antiplane_fe(img, f['mu'], m['mu'])
        Cfe = fe.plane_strain_fe(img, f['lam'], f['mu'], m['lam'], m['mu'])
        kf, Gf_, _ = fs.transverse_moduli(Cfe); t_fe = time.time() - t
        rec['levels'][str(N)] = dict(fft=[float(np.trace(Ga) / 2), float(G), float(k)],
                                     fe=[float(np.trace(Gfe) / 2), float(Gf_), float(kf)], t_fft=t_fft, t_fe=t_fe)
    with open(outp, 'a') as fo:
        fo.write(json.dumps(rec) + '\n')
    print(i, {N: np.round(np.array(v['fft']) / np.array(v['fe']) - 1, 4).tolist() for N, v in rec['levels'].items()}, flush=True)
