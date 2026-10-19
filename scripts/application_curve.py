"""Curve and cost figures for the application example (section 4.9, Fig. 14).
Refits the parametric mode exactly as scripts/application.py (same draw, same seed), writes the predicted E2/Em-V_f
curve with its 95 % interval, checks that the chosen design reproduces results/application.json, and computes the
offline cost of the 40 training samples and the cost of a high-fidelity bisection with the single-core benchmark times
of section 4.7 (results/cost_benchmark.json).
Usage: python3 -I scripts/application_curve.py   -> results/application_curve.csv, results/application_cost.json
(results/application.json must exist: python3 scripts/application.py)"""
import sys, os, json, math
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from mfhomog import data as D, surrogates as S, design

NH, REP, TARGET = 40, 0, 3.0
CONTRAST, NUF, NUM, SG = 20.0, 0.22, 0.35, 0.5
pool = D.load('data/pool.jsonl'); lfx = D.load('data/lfx.jsonl')
XP, YP, LOP, UPP = D.arrays(pool, 'HF'); _, YP_L, _, _ = D.arrays(pool, 128); XX, YX_L, LOX, UPX = D.arrays(lfx, 128)
rng = np.random.default_rng(10_000 * NH + REP)
iH = rng.choice(len(pool), NH, replace=False); iX = rng.choice(len(lfx), 3 * NH, replace=False)
XL = np.vstack([XP[iH], XX[iX]]); YL = np.vstack([YP_L[iH], YX_L[iX]])
LOL = np.vstack([LOP[iH], LOX[iX]]); UPL = np.vstack([UPP[iH], UPX[iX]])
par = S.RecursiveAR1('logit', REP).fit(XP[iH], YP[iH], LOP[iH], UPP[iH], XL=XL, YL=YL, LOL=LOL, UPL=UPL)


def rec_for(vf):
    x = np.array([vf, np.log10(CONTRAST), NUF, NUM, SG])
    return dict(idx=0, x=list(x), seed=0, phys=design.to_phys(x), levels={})


vfs = np.linspace(0.20, 0.65, 2001)
recs = [rec_for(v) for v in vfs]
X = np.array([r['x'] for r in recs]); B = [D.bounds(r) for r in recs]
LO = np.array([b[0] for b in B]); UP = np.array([b[1] for b in B])
Yp, lo, up = par.predict(X, LO, UP)
e2 = lambda Y: np.array([D.engineering_from(r, y)[0] for r, y in zip(recs, Y)])
E2, E2lo, E2up = e2(Yp), e2(lo), e2(up)
# bounds on E2 (Hashin-Shtrikman lower/upper of k23, G23, G12 mapped through the same engineering relations)
E2L, E2U = e2(LO), e2(UP)
pd.DataFrame(dict(vf=vfs, E2=E2, E2_lo95=E2lo, E2_up95=E2up, E2_HSlow=E2L, E2_HSup=E2U)).to_csv(
    'results/application_curve.csv', index=False)
A = json.load(open('results/application.json'))
k = int(np.argmin(np.abs(E2 - TARGET)))
# same design point; E2 agrees to round-off of the GP hyperparameter optimisation (about 1e-8 relative)
assert abs(vfs[k] - A['vf_star']) < 1e-12 and abs(E2[k] / A['E2_param'] - 1) < 1e-6, 'refit does not reproduce application.json'
Bm = json.load(open('results/cost_benchmark.json'))['summary']
t = {kk: Bm[kk]['median'] for kk in ('128', '256', '512', 'geometry')}
tHF = t['256'] + t['512'] + t['geometry']; cLF = t['128'] + t['geometry']
n_bis = math.ceil(math.log2((0.65 - 0.20) / 0.001))
out = dict(train_data_cpu_min=(NH * tHF + 3 * NH * cLF + NH * t['128']) / 60, hf_cpu_s=tHF,
           hf_scan_2001_cpu_h=2001 * tHF / 3600, bisection_steps_dvf_0p001=n_bis, bisection_hf_cpu_min=n_bis * tHF / 60,
           E2_range_on_curve=[float(E2.min()), float(E2.max())], interval_width_pct_median=float(np.median((E2up - E2lo) / E2) * 100),
           vf_star=float(vfs[k]))
# Mori-Tanaka comparison and the check on the 8 arrangements (numbers quoted in section 4.9)
hf, hy = np.array(A['hf_E2']), np.array(A['hybrid_E2']); j = int(np.argmin(np.abs(E2L - TARGET)))
out.update(MT_at_vf_star=float(E2L[k]), MT_vs_hf_mean_pct=float((E2L[k] / hf.mean() - 1) * 100), MT_vf_for_target=float(vfs[j]),
           param_E2_at_MT_vf=float(E2[j]), param_at_MT_vf_vs_target_pct=float((E2[j] / TARGET - 1) * 100),
           hf_range_pct_of_mean=float((hf.max() - hf.min()) / hf.mean() * 100), hf_all_in_interval=bool(np.all((hf >= A['E2_param_95'][0]) & (hf <= A['E2_param_95'][1]))),
           hybrid_dev_mean_pct=float(np.mean(np.abs(hy / hf - 1)) * 100), hybrid_dev_max_pct=float(np.max(np.abs(hy / hf - 1)) * 100),
           hybrid_hf_corr=float(np.corrcoef(hy, hf)[0, 1]))
json.dump(out, open('results/application_cost.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
