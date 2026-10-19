"""Per-sample test predictions for one (nH, repeat) - for parity plots and error maps. Same training draw as
run_budget.py (rng = default_rng(10000 nH + rep), LF:HF = 4, LF level 128). Run only after the final evaluation.
Usage: python3 scripts/predictions.py NH REP METHODS OUT.npz [EVALSET=test]"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mfhomog import data as D, surrogates as S, nn_models as NN

nH, rep, methods, outp = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3].split(','), sys.argv[4]
EV = sys.argv[5] if len(sys.argv) > 5 else 'test'
LF, RATIO = 128, 4
pool = D.load('data/pool.jsonl'); lfx = D.load('data/lfx.jsonl'); ev = D.load(f'data/{EV}.jsonl')
XP, YP, LOP, UPP = D.arrays(pool, 'HF'); _, YP_L, _, _ = D.arrays(pool, LF)
XX, YX_L, LOX, UPX = D.arrays(lfx, LF)
XE, YE, LOE, UPE = D.arrays(ev, 'HF'); _, YE_L, _, _ = D.arrays(ev, LF); _, YE64, _, _ = D.arrays(ev, 64)
E2t = np.array([D.engineering_from(r, y) for r, y in zip(ev, YE)])
HP = json.load(open('results/nn_hparams.json'))
rng = np.random.default_rng(10_000 * nH + rep)
iH = rng.choice(len(pool), nH, replace=False); iX = rng.choice(len(lfx), (RATIO - 1) * nH, replace=False)
XL = np.vstack([XP[iH], XX[iX]]); YL = np.vstack([YP_L[iH], YX_L[iX]])
LOL = np.vstack([LOP[iH], LOX[iX]]); UPL = np.vstack([UPP[iH], UPX[iX]])
out = dict(X=XE, Y=YE, LO=LOE, UP=UPE, LF=YE_L, RICH=2 * YE_L - YE64, iH=iH,
           gap=np.array([r['phys']['gap'] for r in ev]), ENG=E2t, idx=np.array([r['idx'] for r in ev]))


def make(name):
    h = HP.get(name, {}).get(str(nH), {})
    kind = 'logit' if name.startswith('BN') else 'log'
    if name in ('GP', 'BN-GP'): return S.GP(kind, rep)
    if name in ('AR1', 'BN-AR1'): return S.RecursiveAR1(kind, rep)
    if name in ('NARGP', 'BN-NARGP'): return S.MFGP('nargp', kind, rep)
    if name in ('COR', 'BN-COR'): return S.Corrector(kind, rep)
    if name in ('LINCOR', 'BN-LINCOR'): return S.LinearCorrector(kind, rep)
    if name == 'MLP': return NN.MLP(seed=rep, **h)
    if name == 'BMLP': return NN.MLP(seed=rep, bounded=True, **h)
    if name == 'TLNN': return NN.TLNN(seed=rep, **h)
    if name == 'MFNN': return NN.MFNN(seed=rep, **h)
    if name == 'XGB': return NN.XGB(seed=rep, **h)
    if name == 'MFRNP':
        from mfhomog.mfrnp_adapter import MFRNP
        return MFRNP(rep)
    raise ValueError(name)


for name in methods:
    m = make(name).fit(XP[iH], YP[iH], LOP[iH], UPP[iH], XL=XL, YL=YL, LOL=LOL, UPL=UPL, YLH=YP_L[iH])
    Yp, lo, up = m.predict(XE, LOE, UPE, YLq=YE_L)
    out[name] = Yp; out[name + '_lo'] = lo; out[name + '_up'] = up
    out[name + '_eng'] = np.array([D.engineering_from(r, y) for r, y in zip(ev, Yp)])
    print(name, (np.abs(Yp / YE - 1) * 100).mean(0).round(3), flush=True)
np.savez(outp, **out)
