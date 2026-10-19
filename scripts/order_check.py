"""Section 1.4: sensitivity of the high-fidelity error to the assumed convergence order.
For each reference RVE and modulus, y_N = a + b h^p is fitted to N = 128, 256, 512, 1024 (least squares, p free in
[0.3, 3]); the error of y_H = 2 y_512 - y_256 is then measured against the fitted limit a instead of y_ref (p = 1).
Writes results/order_check.json."""
import sys, os, json
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mfhomog import data as D

ROOT = os.path.join(os.path.dirname(__file__), '..')
ref = D.load(os.path.join(ROOT, 'data', 'ref.jsonl'))
coarse = json.load(open(os.path.join(ROOT, 'results', 'ref_coarse.json')))   # N = 64, 128 (written by fig03_verify.py)


def v(r, N):
    L = r['levels']
    if str(N) in L:
        return np.array([L[str(N)]['G12'], L[str(N)]['G23'], L[str(N)]['k23']])
    return np.array(coarse[str(r['idx'])][str(N)])


Ns = np.array([128, 256, 512, 1024]); h = 1.0 / Ns
P, eH_fit, eref_fit, eH_ref, mono = [], [], [], [], []
for r in ref:
    Y = np.array([v(r, N) for N in Ns])
    yH = 2 * v(r, 512) - v(r, 256); yref = 2 * v(r, 1024) - v(r, 512)
    pr, ef, er = [], [], []
    for j in range(3):
        y = Y[:, j]
        res = least_squares(lambda q: q[0] + q[1] * h ** q[2] - y, x0=[yref[j], (y[0] - yref[j]) / h[0], 1.0],
                            bounds=([-np.inf, -np.inf, 0.3], [np.inf, np.inf, 3.0]))
        a, b, p = res.x; pr.append(p); ef.append(abs(yH[j] / a - 1) * 100); er.append(abs(yref[j] / a - 1) * 100)
    P.append(pr); eH_fit.append(ef); eref_fit.append(er); eH_ref.append(np.abs(yH / yref - 1) * 100)
    lev = [64, 128, 256, 512, 1024]
    mono.append([np.all(np.diff([v(r, N)[j] for N in lev]) < 0) for j in range(3)])
P, eH_fit, eref_fit, eH_ref, mono = map(np.array, (P, eH_fit, eref_fit, eH_ref, mono))
out = dict(p_median=np.median(P, 0).tolist(), p_q25=np.percentile(P, 25, 0).tolist(), p_q75=np.percentile(P, 75, 0).tolist(),
           p_min=P.min(0).tolist(), p_max=P.max(0).tolist(),
           HF_err_vs_fit_median=np.median(eH_fit, 0).tolist(), HF_err_vs_fit_max=eH_fit.max(0).tolist(),
           ref_vs_fit_median=np.median(eref_fit, 0).tolist(), ref_vs_fit_max=eref_fit.max(0).tolist(),
           HF_err_vs_ref_median=np.median(eH_ref, 0).tolist(), HF_err_vs_ref_max=eH_ref.max(0).tolist(),
           monotone_decreasing_share=mono.mean(0).tolist(), n=len(ref))
json.dump(out, open(os.path.join(ROOT, 'results', 'order_check.json'), 'w'), indent=1)
print(json.dumps({k: np.round(v, 3).tolist() if isinstance(v, list) else v for k, v in out.items()}, indent=0))
