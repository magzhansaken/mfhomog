"""Surrogate models compared in the study. Common interface:
    m = Model(...); m.fit(XH, YH, LOH, UPH, XL=None, YL=None, LOL=None, UPL=None)
    mean, lo95, up95 = m.predict(X, LO, UP)          # arrays (n, 3) in normalised units
Targets are the three independent effective moduli (G12/Gm, G23/Gm, k23/km)."""
import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
EPS = 1e-4
Z95 = 1.959964


# ----------------------------------------------------------------------------- target transforms
def to_s(Y, LO, UP, kind):
    if kind == 'log':
        return np.log(Y)
    if kind == 'mt':                      # log of the ratio to the HS lower bound (= Mori-Tanaka estimate)
        return np.log(Y / LO)
    t = np.clip((Y - LO) / (UP - LO), EPS, 1 - EPS)
    return np.log(t / (1 - t))


def from_s(S, LO, UP, kind):
    if kind == 'log':
        return np.exp(S)
    if kind == 'mt':
        return LO * np.exp(S)
    return LO + (UP - LO) / (1 + np.exp(-S))


def back(mu, sd, LO, UP, kind, n_mc=400, rng=None):
    """Mean and 95 % interval in physical units from a Gaussian in transformed space (Monte Carlo)."""
    rng = rng or np.random.default_rng(0)
    z = rng.standard_normal((n_mc,) + mu.shape)
    S = mu[None] + sd[None] * z
    Ys = from_s(S, LO[None], UP[None], kind)
    return Ys.mean(0), np.quantile(Ys, 0.025, 0), np.quantile(Ys, 0.975, 0)


# ----------------------------------------------------------------------------- analytical / raw low fidelity
class Analytic:
    def __init__(self, values_fn, name):
        self.values_fn, self.name = values_fn, name

    def fit(self, *a, **k):
        return self

    def predict(self, X, LO, UP, recs=None):
        Y = np.array([self.values_fn(r) for r in recs])
        return Y, Y, Y


# ----------------------------------------------------------------------------- single-fidelity GP (Kriging)
def _sk_gp(D, seed):
    k = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(D), length_scale_bounds=(1e-2, 1e3), nu=2.5) \
        + WhiteKernel(1e-4, (1e-8, 1e-1))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=4, random_state=seed)


def loo_scale(g):
    """Variance-calibration factor of a fitted sklearn GP from its closed-form leave-one-out residuals
    (training data only): s = sqrt(mean(z_i^2)), z_i = [K^-1 y]_i / sqrt([K^-1]_ii)."""
    from scipy.linalg import cho_solve
    Ki = cho_solve((g.L_, True), np.eye(len(g.L_)))
    z = g.alpha_.ravel() / np.sqrt(np.diag(Ki))
    return float(np.sqrt(np.mean(z ** 2)))


def _with_cal(model, mu, sd, LO, UP):
    """Point prediction and raw 95 % interval as before; a LOO-calibrated interval is stored on the model."""
    m, lo, up = back(mu, sd, LO, UP, model.kind)
    sc = np.array(model.scales)[None, :]
    _, model.cal_lo, model.cal_up = back(mu, sd * sc, LO, UP, model.kind)
    return m, lo, up


class GP:
    """Ordinary Kriging-type GP (Matern 5/2, ARD) on log targets, high-fidelity data only."""
    def __init__(self, kind='log', seed=0, name=None):
        self.kind, self.seed = kind, seed
        self.name = name or {'log': 'GP', 'logit': 'BN-GP', 'mt': 'MT-GP'}[kind]

    def fit(self, XH, YH, LOH, UPH, **kw):
        self.sx = StandardScaler().fit(XH); Xs = self.sx.transform(XH)
        S = to_s(YH, LOH, UPH, self.kind)
        self.gps = [_sk_gp(XH.shape[1], self.seed + j).fit(Xs, S[:, j]) for j in range(S.shape[1])]
        self.scales = [loo_scale(g) for g in self.gps]
        return self

    def predict(self, X, LO, UP, **kw):
        Xs = self.sx.transform(X)
        mu = np.zeros((len(X), len(self.gps))); sd = np.zeros_like(mu)
        for j, g in enumerate(self.gps):
            mu[:, j], sd[:, j] = g.predict(Xs, return_std=True)
        return _with_cal(self, mu, sd, LO, UP)


# ----------------------------------------------------------------------------- multi-fidelity GPs (emukit)
class MFGP:
    """Two-level multi-fidelity GP: 'ar1' = linear autoregressive co-kriging (Kennedy & O'Hagan 2000),
    'nargp' = nonlinear autoregressive GP (Perdikaris et al. 2017). kind = 'log' or 'logit' (bound-normalised)."""
    def __init__(self, mode='ar1', kind='log', seed=0, restarts=3, name=None):
        self.mode, self.kind, self.seed, self.restarts = mode, kind, seed, restarts
        self.name = name or {('ar1', 'log'): 'AR1', ('nargp', 'log'): 'NARGP',
                             ('ar1', 'logit'): 'BN-AR1', ('nargp', 'logit'): 'BN-NARGP'}[(mode, kind)]

    def fit(self, XH, YH, LOH, UPH, XL=None, YL=None, LOL=None, UPL=None, **kw):
        import GPy
        from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays
        np.random.seed(self.seed)
        self.sx = StandardScaler().fit(np.vstack([XL, XH]))
        XLs, XHs = self.sx.transform(XL), self.sx.transform(XH)
        SL, SH = to_s(YL, LOL, UPL, self.kind), to_s(YH, LOH, UPH, self.kind)
        self.models = []; self.norm = []
        D = XH.shape[1]
        for j in range(SH.shape[1]):
            m0, s0 = SL[:, j].mean(), SL[:, j].std() + 1e-12
            yl = (SL[:, j:j + 1] - m0) / s0; yh = (SH[:, j:j + 1] - m0) / s0
            X, Y = convert_xy_lists_to_arrays([XLs, XHs], [yl, yh])
            if self.mode == 'ar1':
                from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel
                from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
                from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
                kern = LinearMultiFidelityKernel([GPy.kern.Matern52(D, ARD=True), GPy.kern.Matern52(D, ARD=True)])
                g = GPyLinearMultiFidelityModel(X, Y, kern, n_fidelities=2)
                g.mixed_noise.Gaussian_noise.variance.constrain_bounded(1e-8, 1e-1, warning=False)
                g.mixed_noise.Gaussian_noise_1.variance.constrain_bounded(1e-8, 1e-1, warning=False)
                w = GPyMultiOutputWrapper(g, 2, n_optimization_restarts=self.restarts, verbose_optimization=False)
                w.optimize()
                self.models.append(w)
            else:
                from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import (
                    make_non_linear_kernels, NonLinearMultiFidelityModel)
                kerns = make_non_linear_kernels(GPy.kern.Matern52, 2, D)
                w = NonLinearMultiFidelityModel(X, Y, n_fidelities=2, kernels=kerns, verbose=False,
                                                optimization_restarts=self.restarts)
                for mm_ in w.models:
                    mm_.Gaussian_noise.variance.constrain_bounded(1e-8, 1e-1, warning=False)
                w.optimize()
                self.models.append(w)
            self.norm.append((m0, s0))
        return self

    def predict(self, X, LO, UP, **kw):
        Xs = self.sx.transform(X)
        Xf = np.hstack([Xs, np.ones((len(X), 1))])  # fidelity index 1 = high
        mu = np.zeros((len(X), len(self.models))); sd = np.zeros_like(mu)
        for j, w in enumerate(self.models):
            m, v = w.predict(Xf)
            m0, s0 = self.norm[j]
            mu[:, j] = m[:, 0] * s0 + m0; sd[:, j] = np.sqrt(np.maximum(v[:, 0], 1e-12)) * s0
        return back(mu, sd, LO, UP, self.kind)


class RecursiveAR1:
    """Recursive auto-regressive co-kriging (Le Gratiet & Garnier 2014): GP_L on the low-fidelity data, then
    y_H(x) = rho * m_L(x) + delta(x), with rho and a constant fitted by generalised least squares together with the
    GP on delta (two fixed-point iterations). Predictive variance = rho^2 var_L + var_delta (nested design)."""
    def __init__(self, kind='log', seed=0, name=None):
        self.kind, self.seed = kind, seed
        self.name = name or ('AR1' if kind == 'log' else 'BN-AR1')

    def fit(self, XH, YH, LOH, UPH, XL=None, YL=None, LOL=None, UPL=None, **kw):
        self.sx = StandardScaler().fit(np.vstack([XL, XH]))
        XLs, XHs = self.sx.transform(XL), self.sx.transform(XH)
        SL, SH = to_s(YL, LOL, UPL, self.kind), to_s(YH, LOH, UPH, self.kind)
        D = XH.shape[1]
        self.gl, self.gd, self.coef = [], [], []
        for j in range(SH.shape[1]):
            gl = _sk_gp(D, self.seed + j).fit(XLs, SL[:, j])
            mL = gl.predict(XHs)
            A = np.column_stack([mL, np.ones_like(mL)])
            beta = np.linalg.lstsq(A, SH[:, j], rcond=None)[0]
            for _ in range(2):
                gd = _sk_gp(D, self.seed + 10 + j).fit(XHs, SH[:, j] - A @ beta)
                K = gd.kernel_(XHs) + 1e-10 * np.eye(len(XHs))
                Ki = np.linalg.solve(K, A)
                beta = np.linalg.solve(A.T @ Ki, Ki.T @ SH[:, j])
            gd = _sk_gp(D, self.seed + 10 + j).fit(XHs, SH[:, j] - A @ beta)
            self.gl.append(gl); self.gd.append(gd); self.coef.append(beta)
        self.dscales = [loo_scale(g) for g in self.gd]
        return self

    def predict(self, X, LO, UP, **kw):
        Xs = self.sx.transform(X)
        mu = np.zeros((len(X), len(self.gl))); sd = np.zeros_like(mu); sdc = np.zeros_like(mu)
        for j, (gl, gd, (rho, c)) in enumerate(zip(self.gl, self.gd, self.coef)):
            mL, sL = gl.predict(Xs, return_std=True); md, sdl = gd.predict(Xs, return_std=True)
            mu[:, j] = rho * mL + c + md; sd[:, j] = np.sqrt(rho ** 2 * sL ** 2 + sdl ** 2)
            sdc[:, j] = np.sqrt(rho ** 2 * sL ** 2 + (self.dscales[j] * sdl) ** 2)   # LOO scale on the delta GP
        m, lo, up = back(mu, sd, LO, UP, self.kind)
        _, self.cal_lo, self.cal_up = back(mu, sdc, LO, UP, self.kind)
        return m, lo, up


class Corrector:
    """Hybrid solver-in-the-loop model: the cheap low-fidelity solve is run for the query microstructure and a GP
    learns the correction to high fidelity, s_H = s_L + d(x, s_L), in the bound-normalised (or log) coordinate.
    Training needs nested data (HF points with their LF values); prediction needs the LF value of the query."""
    def __init__(self, kind='logit', seed=0, name=None, use_sl=True):
        self.kind, self.seed, self.use_sl = kind, seed, use_sl      # use_sl=False: ablation d = d(x) (BN-COR-X)
        self.name = name or ('BN-COR' if kind == 'logit' else 'COR') + ('' if use_sl else '-X')

    def _z(self, X, sl):
        return np.column_stack([X, sl]) if self.use_sl else X

    def fit(self, XH, YH, LOH, UPH, YLH=None, **kw):
        self.SL = to_s(YLH, LOH, UPH, self.kind); SH = to_s(YH, LOH, UPH, self.kind)
        self.sx, self.gps = [], []
        for j in range(SH.shape[1]):
            Z = self._z(XH, self.SL[:, j]); sx = StandardScaler().fit(Z)
            self.gps.append(_sk_gp(Z.shape[1], self.seed + j).fit(sx.transform(Z), SH[:, j] - self.SL[:, j]))
            self.sx.append(sx)
        self.scales = [loo_scale(g) for g in self.gps]
        return self

    def predict(self, X, LO, UP, YLq=None, **kw):
        SLq = to_s(YLq, LO, UP, self.kind)
        mu = np.zeros((len(X), len(self.gps))); sd = np.zeros_like(mu)
        for j, (g, sx) in enumerate(zip(self.gps, self.sx)):
            m, s = g.predict(sx.transform(self._z(X, SLq[:, j])), return_std=True)
            mu[:, j] = SLq[:, j] + m; sd[:, j] = s
        return _with_cal(self, mu, sd, LO, UP)


class LinearCorrector:
    """Ablation of the hybrid model: the GP correction is replaced by a ridge regression of s_H - s_L on
    z = (x, s_L) (ridge parameter by leave-one-out on the training data only). Deterministic."""
    def __init__(self, kind='logit', seed=0, name=None):
        self.kind = kind; self.name = name or ('BN-LINCOR' if kind == 'logit' else 'LINCOR')

    def fit(self, XH, YH, LOH, UPH, YLH=None, **kw):
        from sklearn.linear_model import RidgeCV
        SL = to_s(YLH, LOH, UPH, self.kind); SH = to_s(YH, LOH, UPH, self.kind)
        self.m = []
        for j in range(SH.shape[1]):
            Z = np.column_stack([XH, SL[:, j]]); sx = StandardScaler().fit(Z)
            r = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(sx.transform(Z), SH[:, j] - SL[:, j])
            self.m.append((sx, r))
        return self

    def predict(self, X, LO, UP, YLq=None, **kw):
        SLq = to_s(YLq, LO, UP, self.kind)
        S = np.stack([SLq[:, j] + r.predict(sx.transform(np.column_stack([X, SLq[:, j]]))) for j, (sx, r) in enumerate(self.m)], 1)
        Y = from_s(S, LO, UP, self.kind)
        return Y, Y, Y
