"""Section 4.2: robustness of the main claim to the test sample. Per-sample errors of BN-COR and of the strongest
existing method at the primary budgets are recomputed for all repeats (same draws as run_budget.py), averaged over
repeats and the relative reduction of the mean error is bootstrapped over the test RVEs.
Usage: python3 scripts/bootstrap_test.py NH METHOD OUT.npz   (per-sample errors, one budget and method)
       python3 scripts/bootstrap_test.py summary               (bootstrap; writes results/bootstrap_test.json)"""
import sys, os, json, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
ROOT = os.path.join(os.path.dirname(__file__), '..')
PAIRS = {10: 'BMLP', 20: 'BMLP', 40: 'MFNN'}

if sys.argv[1] != 'summary':
    from mfhomog import data as D, surrogates as S, nn_models as NN
    nH, name, outp = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    pool = D.load('data/pool.jsonl'); lfx = D.load('data/lfx.jsonl'); ev = D.load('data/test.jsonl')
    XP, YP, LOP, UPP = D.arrays(pool, 'HF'); _, YP_L, _, _ = D.arrays(pool, 128)
    XX, YX_L, LOX, UPX = D.arrays(lfx, 128)
    XE, YE, LOE, UPE = D.arrays(ev, 'HF'); _, YE_L, _, _ = D.arrays(ev, 128)
    HP = json.load(open('results/nn_hparams.json')); h = HP.get(name, {}).get(str(nH), {})
    E = []
    for rep in range(20):
        rng = np.random.default_rng(10_000 * nH + rep)
        iH = rng.choice(len(pool), nH, replace=False); iX = rng.choice(len(lfx), 3 * nH, replace=False)
        XL = np.vstack([XP[iH], XX[iX]]); YL = np.vstack([YP_L[iH], YX_L[iX]])
        LOL = np.vstack([LOP[iH], LOX[iX]]); UPL = np.vstack([UPP[iH], UPX[iX]])
        m = {'BN-COR': lambda: S.Corrector('logit', rep), 'BMLP': lambda: NN.MLP(seed=rep, bounded=True, **h),
             'MFNN': lambda: NN.MFNN(seed=rep, **h)}[name]()
        m.fit(XP[iH], YP[iH], LOP[iH], UPP[iH], XL=XL, YL=YL, LOL=LOL, UPL=UPL, YLH=YP_L[iH])
        Yp, _, _ = m.predict(XE, LOE, UPE, YLq=YE_L)
        E.append((np.abs(Yp / YE - 1) * 100).mean(1)); print(nH, name, rep, round(E[-1].mean(), 4), flush=True)
    np.savez(outp, E=np.array(E))
else:
    rng = np.random.default_rng(0); out = {}
    for nH, comp in PAIRS.items():
        a = np.load(os.path.join(ROOT, 'results', f'boot_{nH}_BN-COR.npz'))['E']
        b = np.load(os.path.join(ROOT, 'results', f'boot_{nH}_{comp}.npz'))['E']
        ea, eb = a.mean(0), b.mean(0); n = len(ea)
        red = 1 - ea.mean() / eb.mean()
        idx = rng.integers(0, n, (10000, n))
        boot = 1 - ea[idx].mean(1) / eb[idx].mean(1)
        out[nH] = dict(competitor=comp, repeat_means_ours=a.mean(1).round(4).tolist()[:3], mean_ours=float(a.mean()),
                       mean_comp=float(b.mean()), reduction=float(red), ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                       share_samples_lower=float(np.mean(ea < eb)))
    json.dump(out, open(os.path.join(ROOT, 'results', 'bootstrap_test.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))
