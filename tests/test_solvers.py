import numpy as np, time, sys
sys.path.insert(0, '.')
from mfhomog import fft_solver as fs, micromech as mm, rve, fe_check as fe

def test_homogeneous():
    img = np.zeros((32, 32), bool)
    G, _, _ = fs.antiplane_shear(img, 5.0, 2.0)
    assert np.allclose(G, 2.0 * np.eye(2))
    m = mm.phase(3.0, 0.35)
    C, _, _ = fs.plane_strain(img, m['lam'], m['mu'], m['lam'], m['mu'])
    k, G23, _ = fs.transverse_moduli(C)
    assert abs(k - m['k']) < 1e-10 and abs(G23 - m['mu']) < 1e-10

def test_fft_vs_fe_and_bounds():
    X, r = rve.generate(0.5, n_fibres=16, gap=0.05, sweeps=200, seed=3)
    f, m = mm.phase(60.0, 0.22), mm.phase(3.0, 0.35)
    b = mm.hs_bounds(0.5, f, m)
    for N in (64, 128):
        img = rve.rasterise(X, r, N); vf = img.mean()
        t = time.time(); Ga, ita, _ = fs.antiplane_shear(img, f['mu'], m['mu']); ta = time.time() - t
        t = time.time(); Cf, itp, _ = fs.plane_strain(img, f['lam'], f['mu'], m['lam'], m['mu']); tp = time.time() - t
        Ga_fe = fe.antiplane_fe(img, f['mu'], m['mu']); Cfe = fe.plane_strain_fe(img, f['lam'], f['mu'], m['lam'], m['mu'])
        k, G23, an = fs.transverse_moduli(Cf); kfe, G23fe, _ = fs.transverse_moduli(Cfe)
        bb = mm.hs_bounds(vf, f, m)
        print(f"N={N} vf={vf:.4f} G12 fft={np.trace(Ga)/2:.4f} fe={np.trace(Ga_fe)/2:.4f} | k23 fft={k:.4f} fe={kfe:.4f} | G23 fft={G23:.4f} fe={G23fe:.4f} | aniso={an:.3f} | its {ita},{itp} | t {ta:.2f}s,{tp:.2f}s")
        print('   HS G12', [round(x,4) for x in bb['G12']], 'k23', [round(x,4) for x in bb['k23']], 'G23', [round(x,4) for x in bb['G23']])
        assert bb['G12'][0] <= np.trace(Ga)/2 <= bb['G12'][1]
        assert bb['k23'][0] <= k <= bb['k23'][1] and bb['G23'][0] <= G23 <= bb['G23'][1]

if __name__ == '__main__':
    test_homogeneous(); test_fft_vs_fe_and_bounds(); print('ok')
