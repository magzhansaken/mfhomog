"""Budget experiment: every method, every high-fidelity budget, R repeats.
Usage: python3 scripts/run_budget.py EVALSET OUT.csv [--methods a,b] [--budgets 10,20] [--repeats 20] [--ratio 4]
EVALSET = val (development) or test (final, run once). Training data always come from pool (+ lfx for LF)."""
import sys, os, json, time, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mfhomog import data as D, surrogates as S, nn_models as NN

ap = argparse.ArgumentParser()
ap.add_argument('evalset'); ap.add_argument('out')
ap.add_argument('--methods', default='MT,LF,RICH,GP,BN-GP,AR1,NARGP,BN-AR1,BN-NARGP,MLP,BMLP,XGB,MFNN,TLNN')
ap.add_argument('--budgets', default='10,20,40,80,160')
ap.add_argument('--repeats', type=int, default=20)
ap.add_argument('--ratio', type=int, default=4)
ap.add_argument('--hp', default='results/nn_hparams.json')
ap.add_argument('--first_repeat', type=int, default=0)
ap.add_argument('--lf', type=int, default=64)
a = ap.parse_args()

pool = D.load('data/pool.jsonl'); lfx = D.load('data/lfx.jsonl')
ev = D.load(f'data/{a.evalset}.jsonl')
XP, YP, LOP, UPP = D.arrays(pool, 'HF')
_, YP_L, _, _ = D.arrays(pool, a.lf)
XX, YX_L, LOX, UPX = D.arrays(lfx, a.lf)
XE, YE, LOE, UPE = D.arrays(ev, 'HF')
_, YE_L, _, _ = D.arrays(ev, a.lf); _, YE64, _, _ = D.arrays(ev, 64); _, YE_M, _, _ = D.arrays(ev, 128)
cQ = D.cost(ev, a.lf).mean()  # cost of one low-fidelity solve at query time (hybrid models)
cHF = D.cost(pool, 'HF')
HP = json.load(open(a.hp)) if os.path.exists(a.hp) else {}


def make(name, nH, seed):
    h = HP.get(name, {}).get(str(nH), {})
    if name == 'GP': return S.GP('log', seed)
    if name == 'BN-GP': return S.GP('logit', seed)
    if name == 'MT-GP': return S.GP('mt', seed)       # post-hoc ablation: GP on log(y / Mori-Tanaka)
    if name in ('AR1', 'BN-AR1'):
        return S.RecursiveAR1('logit' if name.startswith('BN') else 'log', seed)
    if name in ('NARGP', 'BN-NARGP'):
        return S.MFGP('nargp', 'logit' if name.startswith('BN') else 'log', seed)
    if name in ('COR', 'BN-COR'):
        return S.Corrector('logit' if name.startswith('BN') else 'log', seed)
    if name == 'BN-COR-X': return S.Corrector('logit', seed, use_sl=False)   # post-hoc ablation: d = d(x)
    if name in ('LINCOR', 'BN-LINCOR'):
        return S.LinearCorrector('logit' if name.startswith('BN') else 'log', seed)
    if name == 'MFRNP':
        from mfhomog.mfrnp_adapter import MFRNP
        return MFRNP(seed)
    if name == 'MLP': return NN.MLP(seed=seed, **h)
    if name == 'BMLP': return NN.MLP(seed=seed, bounded=True, **h)
    if name == 'TLNN': return NN.TLNN(seed=seed, **h)
    if name == 'VRN': return NN.VRN(seed=seed, **h)       # post-hoc comparison baseline (PROTOCOL 15:45)
    if name == 'MFNN': return NN.MFNN(seed=seed, **h)
    if name == 'XGB': return NN.XGB(seed=seed, **h)
    raise ValueError(name)


def metrics(Yp, lo, up, Y, LO, UP, clo=None, cup=None):
    ape = np.abs(Yp / Y - 1) * 100
    out = {}
    for j, t in enumerate(D.TARGETS):
        out[f'mape_{t}'] = ape[:, j].mean(); out[f'max_{t}'] = ape[:, j].max()
        out[f'rmse_{t}'] = np.sqrt(np.mean((Yp[:, j] - Y[:, j]) ** 2))
        out[f'r2_{t}'] = 1 - np.sum((Yp[:, j] - Y[:, j]) ** 2) / np.sum((Y[:, j] - Y[:, j].mean()) ** 2)
        out[f'viol_{t}'] = np.mean((Yp[:, j] < LO[:, j] - 1e-9) | (Yp[:, j] > UP[:, j] + 1e-9)) * 100
        prob = not np.allclose(lo, up)
        out[f'cov95_{t}'] = np.mean((Y[:, j] >= lo[:, j]) & (Y[:, j] <= up[:, j])) * 100 if prob else np.nan
        out[f'width95_{t}'] = np.mean((up[:, j] - lo[:, j]) / Y[:, j]) * 100 if prob else np.nan
        # LOO-calibrated interval (GP-based models only; amendment of 23:20, before any test result)
        out[f'cov95cal_{t}'] = np.mean((Y[:, j] >= clo[:, j]) & (Y[:, j] <= cup[:, j])) * 100 if clo is not None else np.nan
        out[f'width95cal_{t}'] = np.mean((cup[:, j] - clo[:, j]) / Y[:, j]) * 100 if clo is not None else np.nan
    # derived engineering constants via Hill's exact connections
    Ep = np.array([D.engineering_from(r, y) for r, y in zip(ev, Yp)])
    Et = np.array([D.engineering_from(r, y) for r, y in zip(ev, Y)])
    for j, t in enumerate(['E2', 'nu23', 'E1', 'nu12']):
        e = np.abs(Ep[:, j] / Et[:, j] - 1) * 100
        out[f'mape_{t}'] = e.mean(); out[f'max_{t}'] = e.max()
    out['mape_mean3'] = np.mean([out[f'mape_{t}'] for t in D.TARGETS])
    return out


rows = []
methods = a.methods.split(',')
for nH in [int(b) for b in a.budgets.split(',')]:
    for rep in range(a.first_repeat, a.repeats):
        rng = np.random.default_rng(10_000 * nH + rep)
        iH = rng.choice(len(pool), nH, replace=False)
        nL_extra = (a.ratio - 1) * nH
        iX = rng.choice(len(lfx), nL_extra, replace=False)
        XL = np.vstack([XP[iH], XX[iX]]); YL = np.vstack([YP_L[iH], YX_L[iX]])
        LOL = np.vstack([LOP[iH], LOX[iX]]); UPL = np.vstack([UPP[iH], UPX[iX]])
        data_cost = cHF[iH].sum() + np.concatenate([D.cost(pool, a.lf)[iH], D.cost(lfx, a.lf)[iX]]).sum()
        for name in methods:
            t0 = time.time()
            if name in ('MT', 'HT'):
                Yp = np.array([D.analytic(r, name) for r in ev]); lo = up = Yp; fit_t = 0.0; pred_t = 0.0; clo = cup = None
            elif name == 'LF':
                Yp = YE_L.copy(); lo = up = Yp; fit_t = 0.0; pred_t = 0.0; clo = cup = None
            elif name == 'RICH':   # Richardson extrapolation from the 64 and 128 solves of the query (numerical only)
                Yp = 2 * YE_M - YE64; lo = up = Yp; fit_t = 0.0; pred_t = 0.0; clo = cup = None
            else:
                m = make(name, nH, rep)
                m.fit(XP[iH], YP[iH], LOP[iH], UPP[iH], XL=XL, YL=YL, LOL=LOL, UPL=UPL, YLH=YP_L[iH])
                fit_t = time.time() - t0
                t1 = time.time()
                Yp, lo, up = m.predict(XE, LOE, UPE, YLq=YE_L)
                pred_t = (time.time() - t1) / len(XE)
                clo, cup = getattr(m, 'cal_lo', None), getattr(m, 'cal_up', None)
            mf = name in ('AR1', 'NARGP', 'BN-AR1', 'BN-NARGP', 'MFNN', 'TLNN', 'MFRNP')
            hyb = name in ('COR', 'BN-COR', 'BN-COR-X', 'LF', 'RICH', 'LINCOR', 'BN-LINCOR')
            cor = name in ('COR', 'BN-COR', 'BN-COR-X', 'LINCOR', 'BN-LINCOR')   # training also needs the low-fidelity solve at the HF points
            dcost = data_cost if mf else (cHF[iH].sum() + D.cost(pool, a.lf)[iH].sum() if cor else cHF[iH].sum())
            row = dict(method=name, lf=a.lf, nH=nH, nL=len(XL) if mf else 0, rep=rep, fit_s=fit_t, pred_s=pred_t,
                       query_solve=int(hyb), data_cpu_s=dcost if name not in ('MT', 'HT', 'LF', 'RICH') else 0.0,
                       **metrics(Yp, lo, up, YE, LOE, UPE, clo, cup))
            rows.append(row)
            pd.DataFrame([row]).to_csv(a.out, mode='a', header=not os.path.exists(a.out), index=False)
        print(f'nH={nH} rep={rep} done', flush=True)
