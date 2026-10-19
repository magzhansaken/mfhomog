"""Section 4.5: split-conformal calibration of the hybrid (BN-COR) intervals with the validation set.
Same training draws as run_budget.py; normalised scores in the bound-normalised coordinate.
Usage: python3 scripts/conformal.py  -> results/conformal.json"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mfhomog import data as D, surrogates as S

pool = D.load('data/pool.jsonl'); val = D.load('data/val.jsonl'); test = D.load('data/test.jsonl')
XP, YP, LOP, UPP = D.arrays(pool, 'HF'); _, YP_L, _, _ = D.arrays(pool, 128)
sets = {}
for name, recs in (('val', val), ('test', test)):
    X, Y, LO, UP = D.arrays(recs, 'HF'); _, YL, _, _ = D.arrays(recs, 128); sets[name] = (X, Y, LO, UP, YL)


def mu_sd(m, X, LO, UP, YL):
    SL = S.to_s(YL, LO, UP, m.kind); mu = np.zeros_like(SL); sd = np.zeros_like(SL)
    for j, (g, sx) in enumerate(zip(m.gps, m.sx)):
        a, b = g.predict(sx.transform(m._z(X, SL[:, j])), return_std=True); mu[:, j] = SL[:, j] + a; sd[:, j] = b
    return mu, sd


out = {}
for nH, R in ((10, 20), (20, 20), (40, 20), (80, 10), (160, 10)):
    cov_raw, cov_cf, w_raw, w_cf, qs = [], [], [], [], []
    for rep in range(R):
        rng = np.random.default_rng(10_000 * nH + rep)
        iH = rng.choice(len(pool), nH, replace=False)
        m = S.Corrector('logit', rep).fit(XP[iH], YP[iH], LOP[iH], UPP[iH], YLH=YP_L[iH])
        Xv, Yv, LOv, UPv, YLv = sets['val']; mv, sv = mu_sd(m, Xv, LOv, UPv, YLv)
        sc = np.abs(S.to_s(Yv, LOv, UPv, 'logit') - mv) / sv
        n = len(sc); k = int(np.ceil(0.95 * (n + 1))) - 1
        q = np.sort(sc, 0)[min(k, n - 1)]; qs.append(q)
        Xt, Yt, LOt, UPt, YLt = sets['test']; mt, st = mu_sd(m, Xt, LOt, UPt, YLt)
        for z, cv, wl in ((1.959964, cov_raw, w_raw), (q[None, :], cov_cf, w_cf)):
            lo = S.from_s(mt - z * st, LOt, UPt, 'logit'); up = S.from_s(mt + z * st, LOt, UPt, 'logit')
            cv.append(np.mean((Yt >= lo) & (Yt <= up), 0) * 100); wl.append(np.mean((up - lo) / Yt, 0) * 100)
    out[nH] = dict(cov_raw=np.mean(cov_raw, 0).round(2).tolist(), cov_conformal=np.mean(cov_cf, 0).round(2).tolist(),
                   width_raw=np.mean(w_raw, 0).round(3).tolist(), width_conformal=np.mean(w_cf, 0).round(3).tolist(),
                   q_mean=np.mean(qs, 0).round(2).tolist(), cov_conformal_mean3=float(np.mean(cov_cf)),
                   width_conformal_mean3=float(np.mean(w_cf)), cov_raw_mean3=float(np.mean(cov_raw)))
    print(nH, out[nH], flush=True)
json.dump(out, open('results/conformal.json', 'w'), indent=1)
