"""Structure check of the generated fibre arrangements (section 1.1): are they disordered (liquid-like) rather than
lattice-like? For every RVE of the training pool the generator of mfhomog/rve.py is re-run with the stored seed while
the unwrapped displacements are tracked. Reported per RVE:
  - |Psi6|: global hexatic order parameter, |mean_j psi6_j|, psi6_j = mean over the 6 nearest neighbours (periodic) of
    exp(6 i theta_jk); 1 for a perfect hexagonal lattice, O(N^-1/2) for an uncorrelated liquid of N disks;
  - <|psi6_j|>: mean local bond-orientational order;
  - RMS displacement from the initial lattice site in units of the mean nearest-neighbour spacing (> 1 means the
    arrangement has forgotten its starting lattice).
The same quantities are given for the initial lattice, a perfect hexagonal arrangement and 36 uniformly random points
(2000 draws) for reference.
Usage: python3 -I scripts/order_parameter.py results/order_parameter.json"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mfhomog import data as D, rve


def generate_tracked(vf, n_fibres, gap, sweeps, seed):
    """Copy of rve.generate (identical random stream) that also returns the initial positions and the unwrapped moves."""
    rng = np.random.default_rng(seed)
    r = np.sqrt(vf / (n_fibres * np.pi)); dmin = 2 * r * (1 + gap)
    nx = int(np.ceil(np.sqrt(n_fibres))); ny = int(np.ceil(n_fibres / nx)); pts = []
    for j in range(ny):
        for i in range(nx):
            if len(pts) < n_fibres:
                pts.append([(i + 0.5 * (j % 2) + 0.25) / nx, (j + 0.5) / ny])
    X = np.array(pts) % 1.0; X0 = X.copy(); U = np.zeros_like(X)
    step = max(0.5 * (1.0 / np.sqrt(n_fibres) - dmin), 0.05 * r)
    acc = tot = 0
    for s in range(sweeps):
        for i in rng.permutation(n_fibres):
            dx = rng.uniform(-step, step, 2); trial = (X[i] + dx) % 1.0
            d = X - trial; d -= np.round(d); dist = np.sqrt((d ** 2).sum(1)); dist[i] = np.inf
            tot += 1
            if dist.min() >= dmin:
                X[i] = trial; U[i] += dx; acc += 1
        if s % 20 == 19:
            rate = acc / max(tot, 1); step *= 1.2 if rate > 0.45 else (0.8 if rate < 0.35 else 1.0)
            acc = tot = 0
    return X, X0, U, r


def psi6(X):
    n = len(X); d = X[:, None, :] - X[None, :, :]; d -= np.round(d)
    dist = np.sqrt((d ** 2).sum(-1)); np.fill_diagonal(dist, np.inf)
    nb = np.argsort(dist, 1)[:, :6]
    ang = np.arctan2(d[np.arange(n)[:, None], nb, 1], d[np.arange(n)[:, None], nb, 0])
    loc = np.exp(6j * ang).mean(1)
    nn = np.sort(dist, 1)[:, 0].mean()
    return float(np.abs(loc.mean())), float(np.abs(loc).mean()), float(nn)


def main():
    out_path = sys.argv[1]
    pool = D.load('data/pool.jsonl')
    rows = []
    for rec in pool:
        p = rec['phys']
        X, X0, U, r = generate_tracked(p['vf'], 36, p['gap'], 400, rec['seed'])
        Xr, _ = rve.generate(p['vf'], 36, p['gap'], 400, seed=rec['seed'])
        assert np.allclose(X, Xr), 'tracked generator differs from rve.generate'
        g, l, nn = psi6(X); g0, l0, _ = psi6(X0)
        eta = p['vf'] * (1 + p['gap']) ** 2
        rows.append(dict(eta=eta, psi6_global=g, psi6_local=l, rms_disp_over_nn=float(np.sqrt((U ** 2).sum(1).mean()) / nn),
                         psi6_global_init=g0, psi6_local_init=l0))
    # reference: perfect hexagonal arrangement of 36 disks in a periodic 6 x 6 rhombic cell (mapped to a rectangle)
    a = 1 / 6; H = []
    for j in range(6):
        for i in range(6):
            H.append([(i + 0.5 * (j % 2)) * a, j * a * np.sqrt(3) / 2])
    H = np.array(H); H[:, 1] /= (6 * a * np.sqrt(3) / 2)          # periodic in y after scaling (slightly distorted)
    g_hex, l_hex, _ = psi6(H)
    import pandas as pd
    df = pd.DataFrame(rows)
    bins = [0, 0.4, 0.5, 0.6, 0.7]
    df['eta_bin'] = pd.cut(df.eta, bins)
    summ = df.groupby('eta_bin', observed=True)[['psi6_global', 'psi6_local', 'rms_disp_over_nn']].median().round(3)
    res = dict(n=len(df), eta_range=[float(df.eta.min()), float(df.eta.max())],
               psi6_global_median=float(df.psi6_global.median()), psi6_global_p90=float(df.psi6_global.quantile(0.9)),
               psi6_global_max=float(df.psi6_global.max()), psi6_local_median=float(df.psi6_local.median()),
               rms_disp_over_nn_median=float(df.rms_disp_over_nn.median()), rms_disp_over_nn_min=float(df.rms_disp_over_nn.min()),
               init_lattice_psi6_global_median=float(df.psi6_global_init.median()),
               init_lattice_psi6_local_median=float(df.psi6_local_init.median()),
               hexagonal_reference_psi6_global=g_hex, hexagonal_reference_psi6_local=l_hex,
               uncorrelated_scale_1_over_sqrtN=float(1 / np.sqrt(36)),
               uniform_random_36_points_psi6_global_median=float(np.median([psi6(np.random.default_rng(s_).random((36, 2)))[0] for s_ in range(2000)])),
               by_eta_bin={str(k): v for k, v in summ.to_dict('index').items()})
    json.dump(res, open(out_path, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
