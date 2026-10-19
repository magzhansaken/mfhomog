"""Section 4.6: hybrid and parametric modes on the scatter set (10 design points x 8 arrangements), all budgets and
repeats (same training draws as run_budget.py). Incremental: every (n_H, repeat) is appended to
results/scatter_eval.jsonl and skipped on restart. Writes results/scatter_eval.json."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mfhomog import data as D, surrogates as S

pool = D.load('data/pool.jsonl'); lfx = D.load('data/lfx.jsonl'); sc = D.load('data/scatter.jsonl')
XP, YP, LOP, UPP = D.arrays(pool, 'HF'); _, YP_L, _, _ = D.arrays(pool, 128); XX, YX_L, LOX, UPX = D.arrays(lfx, 128)
XS, YS, LOS, UPS = D.arrays(sc, 'HF'); _, YS_L, _, _ = D.arrays(sc, 128)
grp = np.array([r['idx'] for r in sc]) // 8
dev = np.zeros_like(YS)
for g in np.unique(grp):
    dev[grp == g] = YS[grp == g] - YS[grp == g].mean(0)
floor = np.array([np.mean(np.abs(YS[grp == g] / YS[grp == g].mean(0) - 1), 0) for g in np.unique(grp)]).mean(0) * 100
cv = np.array([YS[grp == g].std(0, ddof=1) / YS[grp == g].mean(0) for g in np.unique(grp)]).mean(0) * 100
out = dict(floor=floor.round(3).tolist(), cv=cv.round(3).tolist(), budgets={})
JL = 'results/scatter_eval.jsonl'
done = {}
if os.path.exists(JL):
    for l in open(JL):
        r = json.loads(l); done[(r['nH'], r['rep'])] = r
for nH, R in ((10, 20), (20, 20), (40, 20), (80, 10), (160, 10)):
    res = {'BN-COR': [], 'BN-AR1': [], 'r2w_COR': []}
    for rep in range(R):
        if (nH, rep) in done:
            for k in res:
                res[k].append(np.array(done[(nH, rep)][k]))
            continue
        rng = np.random.default_rng(10_000 * nH + rep)
        iH = rng.choice(len(pool), nH, replace=False); iX = rng.choice(len(lfx), 3 * nH, replace=False)
        XL = np.vstack([XP[iH], XX[iX]]); YL = np.vstack([YP_L[iH], YX_L[iX]])
        LOL = np.vstack([LOP[iH], LOX[iX]]); UPL = np.vstack([UPP[iH], UPX[iX]])
        hyb = S.Corrector('logit', rep).fit(XP[iH], YP[iH], LOP[iH], UPP[iH], YLH=YP_L[iH])
        par = S.RecursiveAR1('logit', rep).fit(XP[iH], YP[iH], LOP[iH], UPP[iH], XL=XL, YL=YL, LOL=LOL, UPL=UPL)
        Yh, _, _ = hyb.predict(XS, LOS, UPS, YLq=YS_L); Yp, _, _ = par.predict(XS, LOS, UPS)
        res['BN-COR'].append(np.mean(np.abs(Yh / YS - 1), 0) * 100); res['BN-AR1'].append(np.mean(np.abs(Yp / YS - 1), 0) * 100)
        res['r2w_COR'].append(1 - np.sum((Yh - YS) ** 2, 0) / np.sum(dev ** 2, 0))
        print(nH, rep, res['BN-COR'][-1].round(3), res['BN-AR1'][-1].round(3), res['r2w_COR'][-1].round(3), flush=True)
        with open(JL, 'a') as fo:
            fo.write(json.dumps(dict(nH=nH, rep=rep, **{k: [float(v) for v in res[k][-1]] for k in res})) + '\n')
    out['budgets'][nH] = {k: dict(mean=np.mean(v, 0).round(3).tolist(), sd=np.std(v, 0, ddof=1).round(3).tolist()) for k, v in res.items()}
    out['budgets'][nH]['frac_reps_COR_below_floor'] = float(np.mean([np.all(e < floor) for e in res['BN-COR']]))
json.dump(out, open('results/scatter_eval.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
