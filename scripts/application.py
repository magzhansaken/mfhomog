"""Application example: choose the fibre volume fraction that gives a target transverse modulus E2 for a glass/epoxy-like
system (E_f/E_m = 20, nu_f = 0.22, nu_m = 0.35, gap parameter s_g = 0.5), with the parametric mode trained on 40 HF samples
(training draw of repeat 0, as in the test runs). The chosen design is then checked on 8 new random arrangements with the
high-fidelity solver and with the hybrid mode. Output: results/application.json"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mfhomog import data as D, surrogates as S, design, rve, fft_solver as fs, micromech as mm

NH, REP, TARGET = 40, 0, 3.0          # target E2 / Em
CONTRAST, NUF, NUM, SG = 20.0, 0.22, 0.35, 0.5
pool = D.load('data/pool.jsonl'); lfx = D.load('data/lfx.jsonl')
XP, YP, LOP, UPP = D.arrays(pool, 'HF'); _, YP_L, _, _ = D.arrays(pool, 128); XX, YX_L, LOX, UPX = D.arrays(lfx, 128)
rng = np.random.default_rng(10_000 * NH + REP)
iH = rng.choice(len(pool), NH, replace=False); iX = rng.choice(len(lfx), 3 * NH, replace=False)
XL = np.vstack([XP[iH], XX[iX]]); YL = np.vstack([YP_L[iH], YX_L[iX]])
LOL = np.vstack([LOP[iH], LOX[iX]]); UPL = np.vstack([UPP[iH], UPX[iX]])
par = S.RecursiveAR1('logit', REP).fit(XP[iH], YP[iH], LOP[iH], UPP[iH], XL=XL, YL=YL, LOL=LOL, UPL=UPL)
hyb = S.Corrector('logit', REP).fit(XP[iH], YP[iH], LOP[iH], UPP[iH], YLH=YP_L[iH])


def rec_for(vf, idx=0, seed=0):
    x = np.array([vf, np.log10(CONTRAST), NUF, NUM, SG])
    return dict(idx=idx, x=list(x), seed=seed, phys=design.to_phys(x), levels={})


def e2(recs, Y):
    return np.array([D.engineering_from(r, y)[0] for r, y in zip(recs, Y)])


# 1) scan with the parametric mode
vfs = np.linspace(0.20, 0.65, 2001)
recs = [rec_for(v) for v in vfs]
X = np.array([r['x'] for r in recs]); B = [D.bounds(r) for r in recs]
LO = np.array([b[0] for b in B]); UP = np.array([b[1] for b in B])
t0 = time.process_time(); Yp, lo, up = par.predict(X, LO, UP); t_scan = time.process_time() - t0
E2 = e2(recs, Yp); k = int(np.argmin(np.abs(E2 - TARGET))); vf_star = float(vfs[k])
E2_lo, E2_up = float(e2([recs[k]], lo[k:k + 1])[0]), float(e2([recs[k]], up[k:k + 1])[0])
# 2) check on 8 new arrangements at the chosen design (high fidelity and hybrid)
p = design.to_phys(np.array([vf_star, np.log10(CONTRAST), NUF, NUM, SG]))
m, f = mm.phase(1.0, NUM), mm.phase(CONTRAST, NUF)
hf, hy, t_hf, t_hy = [], [], [], []
for s in range(8):
    t0 = time.process_time()
    Xc, r = rve.generate(vf_star, 36, p['gap'], 400, seed=7_000_000 + s); t_geo = time.process_time() - t0
    rec = rec_for(vf_star, s, 7_000_000 + s)
    for N in (128, 256, 512):
        t0 = time.process_time()
        img = rve.rasterise(Xc, r, N)
        Ga, _, _ = fs.antiplane_shear(img, f['mu'], m['mu'], tol=1e-5)
        C, _, _ = fs.plane_strain(img, f['lam'], f['mu'], m['lam'], m['mu'], tol=1e-5)
        kk, G, _ = fs.transverse_moduli(C)
        rec['levels'][str(N)] = dict(G12=float(np.trace(Ga) / 2), G23=float(G), k23=float(kk), t=time.process_time() - t0)
    yH = D.normalised(rec, 'HF'); yL = D.normalised(rec, 128); lo1, up1 = D.bounds(rec)
    yh, _, _ = hyb.predict(np.array([rec['x']]), lo1[None], up1[None], YLq=yL[None])
    hf.append(D.engineering_from(rec, yH)[0]); hy.append(D.engineering_from(rec, yh[0])[0])
    t_hf.append(t_geo + rec['levels']['256']['t'] + rec['levels']['512']['t']); t_hy.append(t_geo + rec['levels']['128']['t'])
hf, hy = np.array(hf), np.array(hy)
out = dict(target=TARGET, vf_star=vf_star, E2_param=float(E2[k]), E2_param_95=[E2_lo, E2_up], scan_points=len(vfs),
           scan_cpu_s=t_scan, hf_E2=hf.tolist(), hf_mean=float(hf.mean()), hf_std=float(hf.std(ddof=1)),
           hf_dev_from_target_pct=float((hf.mean() / TARGET - 1) * 100), hybrid_E2=hy.tolist(),
           hybrid_mape_vs_hf_pct=float(np.mean(np.abs(hy / hf - 1)) * 100), hf_cpu_s_median=float(np.median(t_hf)),
           hybrid_cpu_s_median=float(np.median(t_hy)), equivalent_hf_scan_cpu_h=float(len(vfs) * np.median(t_hf) / 3600))
json.dump(out, open('results/application.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
