"""Sections 1.3 and 2.1: facts about the HS bounds and the coarse-grid (N = 128) error on the generated data
(training pool and test set; no model involved).
Writes results/bounds_stats.json."""
import sys, os, json, itertools
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mfhomog import data as D, micromech as mm

ROOT = os.path.join(os.path.dirname(__file__), '..')
T = ['G12', 'G23', 'k23']
out = {}

# 1) well-ordered phases over the whole design box (corners are the extremes of k_f/k_m and G_f/G_m)
rk, rg = [], []
for ef, nf, nm in itertools.product([5.0, 100.0], [0.15, 0.30], [0.30, 0.42]):
    f, m = mm.phase(ef, nf), mm.phase(1.0, nm)
    rk.append(f['k'] / m['k']); rg.append(f['mu'] / m['mu'])
out['min_kf_over_km'] = min(rk); out['min_Gf_over_Gm'] = min(rg)

# 2) Halpin-Tsai with the Halpin-Kardos factors equals the HS lower bound (max relative difference on a grid)
dmax = dmt = 0.0
for vf, ef, nf, nm in itertools.product(np.linspace(0.2, 0.65, 10), np.geomspace(5, 100, 10), [0.15, 0.3], [0.3, 0.42]):
    f, m = mm.phase(ef, nf), mm.phase(1.0, nm)
    ht, mt = mm.halpin_tsai_set(vf, f, m), mm.mori_tanaka(vf, f, m)
    dmax = max(dmax, max(abs(ht[t] / mt[t] - 1) for t in T))
    me = mm.mori_tanaka_eshelby(vf, f, m)   # independent Eshelby-tensor computation of Mori-Tanaka
    dmt = max(dmt, max(abs(me[t] / mt[t] - 1) for t in T))
out['max_rel_diff_HT_vs_HSlower'] = dmax
out['max_rel_diff_MTeshelby_vs_HSlower'] = dmt

# 3) data inside the bounds; relative position t = (y - L)/(U - L); relative width (U - L)/L
for name in ('pool', 'test'):
    recs = D.load(os.path.join(ROOT, 'data', f'{name}.jsonl'))
    X, Y, LO, UP = D.arrays(recs, 'HF')
    o = {'n': len(recs)}
    for lev in (64, 128, 256, 512, 'HF'):
        _, Yl, _, _ = D.arrays(recs, lev)
        o[f'inside_{lev}'] = int(np.sum(np.all((Yl > LO) & (Yl < UP), 1)))
    t = (Y - LO) / (UP - LO); w = (UP - LO) / LO
    vf, lc = X[:, 0], X[:, 1]
    for k, tn in enumerate(T):
        o[tn] = dict(t_median=float(np.median(t[:, k])), t_min=float(t[:, k].min()), t_max=float(t[:, k].max()),
                     width_median=float(np.median(w[:, k])), width_max=float(w[:, k].max()),
                     rho_t_vf=float(spearmanr(t[:, k], vf)[0]), rho_t_lgEf=float(spearmanr(t[:, k], lc)[0]),
                     mt_err_median=float(np.median(np.abs(LO[:, k] / Y[:, k] - 1)) * 100),
                     mt_err_max=float(np.max(np.abs(LO[:, k] / Y[:, k] - 1)) * 100),
                     mt_mape=float(np.mean(np.abs(LO[:, k] / Y[:, k] - 1)) * 100))
    _, YL, _, _ = D.arrays(recs, 128); rel = (YL / Y - 1) * 100       # coarse-grid error relative to HF (section 2.1)
    o['coarse128'] = dict(rel_median=np.median(rel, 0).tolist(), rel_max=rel.max(0).tolist(), n_below_HF=(rel < 0).sum(0).tolist(),
                          modulus_max_over_min=(Y.max(0) / Y.min(0)).tolist())
    out[name] = o
json.dump(out, open(os.path.join(ROOT, 'results', 'bounds_stats.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
