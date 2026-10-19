"""External dataset 2: short natural-fibre composites (3D FE homogenisation, CC BY 4.0).
Source: github.com/roysondsouza316/natural-fibre-rve-dataset-surrogate (dataset.csv, 2520 rows; independent
test set of 30 RVEs / 60 property sets). Only one fidelity is available, so this set tests the physics-prior
and bound-normalisation parts of the method.
Usage: python3 scripts/run_external_nf.py budgets OUT.csv   |   python3 scripts/run_external_nf.py final OUT.csv"""
import sys, os, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mfhomog import surrogates as S, nn_models as NN, micromech as mm

ROOT = os.path.join(os.environ.get('EXT_DIR', 'external'), 'nf', 'src', 'out')   # scripts/get_external.sh
D_THICK = 10.0  # fibre thickness, micrometres (README)
FEAT = ['fiber_fraction', 'Ef_Em', 'width_to_thickness_ratio', 'logL', 'tdeg', 'dist']
# waviness enters as the mean deviation inclination (degrees), as in the dataset authors' own surrogate
# (01_train_surrogates.py: theta_deg(kappa) = arccos(coth kappa - 1/kappa), 0 for straight fibres)


def theta_deg(kappa):
    return 0.0 if kappa == 0 else float(np.degrees(np.arccos(1 / np.tanh(kappa) - 1 / kappa)))
TGT = ['E3', 'E12', 'G13', 'G12']   # E12 = mean(E1, E2) transverse; G13 = mean(G13, G23)


def prep(d):
    d = d.copy()
    d['dist'] = (d['length_distribution_type'] == 'exponential').astype(float)
    d['E3'] = d['E3_Em']; d['E12'] = 0.5 * (d['E1_Em'] + d['E2_Em'])
    d['G13'] = 0.5 * (d['G13_Em'] + d['G23_Em']); d['G12'] = d['G12_Em']
    d['grp'] = d[['fiber_fraction', 'width_to_thickness_ratio', 'length_value', 'theta', 'dist']].astype(str).agg('|'.join, axis=1)
    d['logL'] = np.log10(d['length_value']); d['tdeg'] = d['theta'].map(lambda k: theta_deg(int(k)))
    return d


def prep_test():
    """Independent FE test set (30 RVEs x 2 contrasts = 60 rows) with the design inputs of the dataset authors
    (test_set_design.csv) and the published predictions of their GP of record trained on all 2520 rows."""
    des = pd.read_csv(f'{ROOT}/test_set/test_set_design.csv')
    fe = pd.read_csv(f'{ROOT}/test_set/test_set_fe_results.csv')
    for t in ['E1', 'E2', 'E3', 'G12', 'G13', 'G23']:
        fe[t + '_Em'] = fe[t] / fe['matrix_modulus']
    fe = fe.drop(columns=['dist']).merge(des[['sim_name', 'dist', 'vf', 'alpha', 'Lp', 'tdeg']], left_on='sim', right_on='sim_name')
    t = pd.DataFrame(dict(fiber_fraction=fe['vf'], Ef_Em=fe['fiber_modulus'] / fe['matrix_modulus'],
                          width_to_thickness_ratio=fe['alpha'], length_value=fe['Lp'], logL=np.log10(fe['Lp']),
                          tdeg=fe['tdeg'], length_distribution_type=fe['dist'], sim=fe['sim']))
    for c in ['E1_Em', 'E2_Em', 'E3_Em', 'G12_Em', 'G13_Em', 'G23_Em']:
        t[c] = fe[c].values
    t['dist'] = (t['length_distribution_type'] == 'exponential').astype(float)
    t['E3'] = t['E3_Em']; t['E12'] = 0.5 * (t['E1_Em'] + t['E2_Em'])
    t['G13'] = 0.5 * (t['G13_Em'] + t['G23_Em']); t['G12'] = t['G12_Em']
    sp = pd.read_csv(f'{ROOT}/test_set/test_set_surrogate_predictions.csv')
    sp['EfEm_r'] = sp['EfEm'].round(2); t['EfEm_r'] = t['Ef_Em'].round(2)
    sp = sp[['sim_name', 'EfEm_r'] + [c for c in sp.columns if c.endswith('_gp')]]
    t = t.merge(sp, left_on=['sim', 'EfEm_r'], right_on=['sim_name', 'EfEm_r'], how='left')
    assert t['E3_Em_gp'].notna().all() and len(t) == 60
    pub = np.stack([t['E3_Em_gp'], 0.5 * (t['E1_Em_gp'] + t['E2_Em_gp']), 0.5 * (t['G13_Em_gp'] + t['G23_Em_gp']), t['G12_Em_gp']], 1)
    return t, pub


def physics(d):
    vf, ef = d['fiber_fraction'].values, d['Ef_Em'].values
    deq = D_THICK * np.sqrt(d['width_to_thickness_ratio'].values)
    reuss = 1 / (vf / ef + (1 - vf)); voigt = vf * ef + (1 - vf)
    # Voigt/Reuss for every modulus (equal Poisson ratios assumed: G ratios = E ratios); shear normalised by Em
    g = 1 / (2 * 1.35)  # Gm/Em for nu_m = 0.35; only a scale, cancels in the bound-normalised coordinate
    LO = np.stack([reuss, reuss, reuss * g, reuss * g], 1); UP = np.stack([voigt, voigt, voigt * g, voigt * g], 1)
    ht3 = mm.halpin_tsai(ef, 1.0, vf, 2 * d['length_value'].values / deq)
    ht1 = mm.halpin_tsai(ef, 1.0, vf, 2.0)
    htg = mm.halpin_tsai(ef, 1.0, vf, 1.0) * g
    HT = np.stack([ht3, ht1, htg, htg], 1)
    return LO, UP, HT


class Prior:
    """GP on the log-ratio to the Halpin-Tsai prior (physics prior mean)."""
    def __init__(self, seed): self.seed = seed; self.name = 'HT-GP'

    def fit(self, X, Y, HT):
        self.g = S.GP('log', self.seed).fit(X, Y / HT, HT, HT); return self

    def predict(self, X, HT):
        m, lo, up = self.g.predict(X, HT, HT); return m * HT, lo * HT, up * HT


def fit_predict(name, Xtr, Ytr, LOtr, UPtr, HTtr, Xte, LOte, UPte, HTte, seed, nrows):
    if name == 'GP': m = S.GP('log', seed).fit(Xtr, Ytr, LOtr, UPtr); return m.predict(Xte, LOte, UPte)
    if name == 'BN-GP': m = S.GP('logit', seed).fit(Xtr, Ytr, LOtr, UPtr); return m.predict(Xte, LOte, UPte)
    if name == 'HT-GP': m = Prior(seed).fit(Xtr, Ytr, HTtr); return m.predict(Xte, HTte)
    if name == 'HT': return HTte, HTte, HTte
    if name == 'MLP': m = NN.MLP(seed=seed, width=32, wd=1e-4, n_ens=3).fit(Xtr, Ytr, LOtr, UPtr); return m.predict(Xte, LOte, UPte)
    if name == 'XGB': m = NN.XGB(seed=seed).fit(Xtr, Ytr, LOtr, UPtr); return m.predict(Xte, LOte, UPte)
    if name == 'POLY3':
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        from sklearn.linear_model import Ridge
        sx = StandardScaler().fit(Xtr); pf = PolynomialFeatures(3)
        A = pf.fit_transform(sx.transform(Xtr)); B = pf.transform(sx.transform(Xte))
        P = np.stack([np.exp(Ridge(1e-6).fit(A, np.log(Ytr[:, j])).predict(B)) for j in range(Ytr.shape[1])], 1)
        return P, P, P
    raise ValueError(name)


def metr(P, lo, up, Y, LO, UP):
    out = {}
    for j, t in enumerate(TGT):
        ape = np.abs(P[:, j] / Y[:, j] - 1) * 100
        out[f'mape_{t}'] = ape.mean(); out[f'max_{t}'] = ape.max()
        out[f'r2_{t}'] = 1 - np.sum((P[:, j] - Y[:, j]) ** 2) / np.sum((Y[:, j] - Y[:, j].mean()) ** 2)
        out[f'cov95_{t}'] = (np.mean((Y[:, j] >= lo[:, j]) & (Y[:, j] <= up[:, j])) * 100) if not np.allclose(lo, up) else np.nan
    out['mape_mean'] = np.mean([out[f'mape_{t}'] for t in TGT])
    return out


def main():
    mode, outp = sys.argv[1], sys.argv[2]
    d = prep(pd.read_csv(f'{ROOT}/dataset.csv'))
    X = d[FEAT].values.astype(float); Y = d[TGT].values; LO, UP, HT = physics(d)
    methods = ['HT', 'GP', 'HT-GP', 'BN-GP', 'MLP', 'XGB', 'POLY3']
    if mode == 'budgets':
        groups = d['grp'].unique()
        for ng in [6, 12, 24, 48, 96]:
            for rep in range(20):
                rng = np.random.default_rng(50_000 + 1000 * ng + rep)
                tr = d['grp'].isin(rng.choice(groups, ng, replace=False)).values; te = ~tr
                for name in methods:
                    if name == 'POLY3' and tr.sum() < 100:
                        continue
                    t0 = time.time()
                    P, lo, up = fit_predict(name, X[tr], Y[tr], LO[tr], UP[tr], HT[tr], X[te], LO[te], UP[te], HT[te], rep, tr.sum())
                    row = dict(method=name, n_groups=ng, n_rows=int(tr.sum()), rep=rep, fit_s=time.time() - t0, **metr(P, lo, up, Y[te], LO[te], UP[te]))
                    pd.DataFrame([row]).to_csv(outp, mode='a', header=not os.path.exists(outp), index=False)
                print(ng, rep, flush=True)
    else:  # final: subsets of the main set -> independent test set (locked until this run)
        t, pub = prep_test()
        Xt = t[FEAT].values.astype(float); Yt = t[TGT].values; LOt, UPt, HTt = physics(t)
        row = dict(method='PUB-GP-2520', n_groups=360, n_rows=len(d), rep=0, fit_s=np.nan, **metr(pub, pub, pub, Yt, LOt, UPt))
        pd.DataFrame([row]).to_csv(outp, mode='a', header=not os.path.exists(outp), index=False)
        groups = d['grp'].unique()
        for ng in [6, 12, 24, 48, 96]:
            for rep in range(20 if ng <= 24 else 10):     # fixed before any result (compute time)
                rng = np.random.default_rng(70_000 + 1000 * ng + rep)
                tr = d['grp'].isin(rng.choice(groups, ng, replace=False)).values
                for name in methods:
                    if name == 'POLY3' and tr.sum() < 100:
                        continue
                    t0 = time.time()
                    P, lo, up = fit_predict(name, X[tr], Y[tr], LO[tr], UP[tr], HT[tr], Xt, LOt, UPt, HTt, rep, tr.sum())
                    row = dict(method=name, n_groups=ng, n_rows=int(tr.sum()), rep=rep, fit_s=time.time() - t0, **metr(P, lo, up, Yt, LOt, UPt))
                    pd.DataFrame([row]).to_csv(outp, mode='a', header=not os.path.exists(outp), index=False)
                print(ng, rep, flush=True)

if __name__ == '__main__':
    main()
