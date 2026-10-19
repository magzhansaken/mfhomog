"""Generate the multi-fidelity dataset (FFT homogenisation at several grid resolutions).

Usage: python3 scripts/generate_data.py SET OUT.jsonl [workers]
SET in {pool, val, test, lfx, scatter, ref}. Each line: inputs, seed, results per resolution, timings.
Seeds: design seed and microstructure seeds differ for every set, so sets never share points.
"""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mfhomog import design, rve, fft_solver as fs, micromech as mm

SETS = {  # n points, design seed, microstructure seed offset, resolutions
    'pool': (400, 101, 1_000_000, (64, 128, 256, 512)),
    'val': (100, 202, 2_000_000, (64, 128, 256, 512)),
    'test': (300, 303, 3_000_000, (64, 128, 256, 512)),
    'lfx': (800, 404, 4_000_000, (64, 128)),     # planned 1600, capped at 800 (PROTOCOL 21:40); same Sobol prefix
    'scatter': (10, 505, 5_000_000, (64, 128, 256, 512)),   # each point x 8 microstructure seeds
    'ref': (12, 606, 6_000_000, (256, 512, 1024)),
}
N_FIBRES = 36
TOL = 1e-5


def solve_point(args):
    idx, x, seed, levels = args
    p = design.to_phys(x)
    m = mm.phase(1.0, p['nu_m']); f = mm.phase(p['contrast'], p['nu_f'])
    t0 = time.time()
    X, r = rve.generate(p['vf'], n_fibres=N_FIBRES, gap=p['gap'], sweeps=400, seed=seed)
    t_geo = time.time() - t0
    out = dict(idx=int(idx), x=[float(v) for v in x], seed=int(seed), phys=p, t_geo=t_geo, levels={})
    for N in levels:
        img = rve.rasterise(X, r, N)
        t = time.time(); Ga, ia, ra = fs.antiplane_shear(img, f['mu'], m['mu'], tol=TOL); ta = time.time() - t
        t = time.time(); C, ip, rp = fs.plane_strain(img, f['lam'], f['mu'], m['lam'], m['mu'], tol=TOL); tp = time.time() - t
        k23, G23, an = fs.transverse_moduli(C)
        out['levels'][str(N)] = dict(G12=float(np.trace(Ga) / 2), k23=float(k23), G23=float(G23), aniso=float(an),
                                     vf_pix=float(img.mean()), its=[int(ia), int(ip)], res=[float(ra), float(rp)],
                                     t=[ta, tp], C=C.tolist(), Ga=Ga.tolist())
    return out


def main():
    name, outp = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    n, dseed, soff, levels = SETS[name]
    X = design.sobol(n, dseed)
    jobs = []
    if name == 'scatter':
        for i, x in enumerate(X):
            for k in range(8):
                jobs.append((i * 8 + k, x, soff + i * 100 + k, levels))
    else:
        jobs = [(i, x, soff + i, levels) for i, x in enumerate(X)]
    done = set()
    if os.path.exists(outp):
        done = {json.loads(l)['idx'] for l in open(outp)}
    jobs = [j for j in jobs if j[0] not in done]
    with Pool(workers) as pool, open(outp, 'a') as fo:
        for res in pool.imap_unordered(solve_point, jobs, chunksize=1):
            fo.write(json.dumps(res) + '\n'); fo.flush()


if __name__ == '__main__':
    main()
