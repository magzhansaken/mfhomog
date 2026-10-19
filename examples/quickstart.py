"""Quick start: train the solver-in-the-loop hybrid model (BN-COR) on 20 high-fidelity samples of the training
pool, then predict the elastic constants of a NEW random microstructure with one coarse-grid FFT solve.

Run from the repository root:  python3 examples/quickstart.py
(about 1 min on one CPU core; the coarse solve at N = 128 takes a few tenths of a second, the N = 256/512
reference solves run at the end only to show the error of the prediction)."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mfhomog import data as D, design, micromech as mm, rve, fft_solver as fs, surrogates as S

# 1. training data: 20 high-fidelity samples of the pool with their coarse-grid (N = 128) solutions
pool = D.load('data/pool.jsonl')
rng = np.random.default_rng(0)
idx = rng.choice(len(pool), 20, replace=False)
train = [pool[i] for i in idx]
XH, YH, LOH, UPH = D.arrays(train, 'HF')          # design variables, HF moduli (normalised), HS bounds
_, YLH, _, _ = D.arrays(train, 128)               # coarse-grid moduli of the same microstructures
model = S.Corrector('logit', seed=0).fit(XH, YH, LOH, UPH, YLH=YLH)

# 2. a new microstructure: glass/epoxy-like phases, V_f = 0.5, random arrangement
x = np.array([0.50, np.log10(20.0), 0.22, 0.35, 0.5])   # (V_f, lg(E_f/E_m), nu_f, nu_m, s_gap)
p = design.to_phys(x); f = mm.phase(p['contrast'], p['nu_f']); m = mm.phase(1.0, p['nu_m'])
X, r = rve.generate(p['vf'], n_fibres=36, gap=p['gap'], sweeps=400, seed=12345)

def solve(N):
    img = rve.rasterise(X, r, N)
    Ga, _, _ = fs.antiplane_shear(img, f['mu'], m['mu'], tol=1e-5)
    C, _, _ = fs.plane_strain(img, f['lam'], f['mu'], m['lam'], m['mu'], tol=1e-5)
    k23, G23, _ = fs.transverse_moduli(C)
    return np.array([np.trace(Ga) / 2 / m['mu'], G23 / m['mu'], k23 / m['k']])   # (G12/Gm, G23/Gm, k23/km)

t0 = time.time(); yL = solve(128); t_coarse = time.time() - t0            # the one online coarse solve
b = mm.hs_bounds(p['vf'], f, m)
LO = np.array([[b['G12'][0] / m['mu'], b['G23'][0] / m['mu'], b['k23'][0] / m['k']]])
UP = np.array([[b['G12'][1] / m['mu'], b['G23'][1] / m['mu'], b['k23'][1] / m['k']]])
mu, lo, up = model.predict(x[None, :], LO, UP, YLq=yL[None, :])

# 3. reference value (Richardson extrapolation of N = 256 and 512) only to show the error
t0 = time.time(); yH = 2 * solve(512) - solve(256); t_hf = time.time() - t0

names = ['G12/Gm', 'G23/Gm', 'k23/km']
print(f'coarse solve {t_coarse:.2f} s, high-fidelity reference {t_hf:.1f} s')
print(f'{"modulus":8s} {"coarse":>8s} {"predicted":>10s} {"95% low":>8s} {"95% up":>8s} {"reference":>10s} {"err %":>7s}')
for j, n in enumerate(names):
    print(f'{n:8s} {yL[j]:8.4f} {mu[0, j]:10.4f} {lo[0, j]:8.4f} {up[0, j]:8.4f} {yH[j]:10.4f} {100 * (mu[0, j] / yH[j] - 1):7.2f}')
eng = D.engineering_from(dict(phys=p), mu[0])
print(f'derived constants: E2/Em = {eng[0]:.3f}, nu23 = {eng[1]:.3f}, E1/Em = {eng[2]:.3f}, nu12 = {eng[3]:.3f}')
