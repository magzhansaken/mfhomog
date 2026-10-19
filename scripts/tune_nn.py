"""Choose hyper-parameters of the neural/tree baselines on the VALIDATION set only (repeat 0 of every budget).
Output: results/nn_hparams.json  {method: {budget: params}}"""
import sys, os, json, itertools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mfhomog import data as D, nn_models as NN

only = sys.argv[1].split(',') if len(sys.argv) > 1 else None
out_path = sys.argv[2] if len(sys.argv) > 2 else 'results/nn_hparams.json'
pool = D.load('data/pool.jsonl'); lfx = D.load('data/lfx.jsonl'); val = D.load('data/val.jsonl')
XP, YP, LOP, UPP = D.arrays(pool, 'HF'); LFLEV = int(os.environ.get('LF', '128'))
_, YPL, _, _ = D.arrays(pool, LFLEV)
XX, YXL, LOX, UPX = D.arrays(lfx, LFLEV); XV, YV, LOV, UPV = D.arrays(val, 'HF')
grids = {
    'MLP': [dict(width=w, wd=d) for w, d in itertools.product([16, 32, 64], [1e-5, 1e-4, 1e-3])],
    'BMLP': [dict(width=w, wd=d) for w, d in itertools.product([16, 32, 64], [1e-5, 1e-4, 1e-3])],
    'TLNN': [dict(width=w, wd=d) for w, d in itertools.product([16, 32, 64], [1e-5, 1e-4, 1e-3])],
    'VRN': [dict(width=w, wd=d) for w, d in itertools.product([16, 32, 64], [1e-5, 1e-4, 1e-3])],
    'MFNN': [dict(width=w, wd=d) for w, d in itertools.product([10, 20, 40], [1e-5, 1e-4, 1e-3])],
    'XGB': [dict(max_depth=m, n_estimators=n, learning_rate=l) for m, n, l in itertools.product([2, 3, 4], [200, 400, 800], [0.03, 0.1])],
}
out = {}
for meth, grid in grids.items():
    if only and meth not in only:
        continue
    out[meth] = {}
    for nH in [10, 20, 40, 80, 160]:
        rng = np.random.default_rng(10_000 * nH + 0)
        iH = rng.choice(len(pool), nH, replace=False)
        iX = rng.choice(len(lfx), 3 * nH, replace=False) if len(lfx) >= 3 * nH else np.arange(len(lfx))
        XL = np.vstack([XP[iH], XX[iX]]); YL = np.vstack([YPL[iH], YXL[iX]])
        LOL = np.vstack([LOP[iH], LOX[iX]]); UPL = np.vstack([UPP[iH], UPX[iX]])
        best = None
        for h in grid:
            if meth == 'MLP': m = NN.MLP(seed=0, n_ens=3, **h)
            elif meth == 'BMLP': m = NN.MLP(seed=0, n_ens=3, bounded=True, **h)
            elif meth == 'TLNN': m = NN.TLNN(seed=0, n_ens=3, **h)
            elif meth == 'VRN': m = NN.VRN(seed=0, n_ens=3, **h)
            elif meth == 'MFNN': m = NN.MFNN(seed=0, n_ens=3, **h)
            else: m = NN.XGB(seed=0, **h)
            m.fit(XP[iH], YP[iH], LOP[iH], UPP[iH], XL=XL, YL=YL, LOL=LOL, UPL=UPL)
            Yp, _, _ = m.predict(XV, LOV, UPV)
            e = float((np.abs(Yp / YV - 1) * 100).mean())
            if best is None or e < best[0]:
                best = (e, h)
        out[meth][str(nH)] = best[1]
        print(meth, nH, best, flush=True)
old = json.load(open(out_path)) if os.path.exists(out_path) else {}
old.update(out)
json.dump(old, open(out_path, 'w'), indent=1)
