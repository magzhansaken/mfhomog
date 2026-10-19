"""Neural-network and tree baselines (CPU, small data).
MLP      : back-propagation network on log targets (the 'BP' surrogate common in JME), ensemble of 5.
BMLP     : bound-preserving MLP, y = L + (U - L) * sigmoid(net) - the output layer idea of the
           Voigt-Reuss Net (Keshav & Fritzen 2026, Int. J. Numer. Methods Eng. 127: e70296) applied with Hashin-Shtrikman bounds to scalar moduli.
VRN      : Voigt-Reuss Net (Keshav & Fritzen 2025/2026) with its own Voigt-Reuss spectral normalisation and loss
           (post-hoc comparison baseline, see results/PROTOCOL.md 15:45).
MFNN     : composite multi-fidelity network of Meng & Karniadakis (2020, J. Comput. Phys. 401, 109020).
TLNN     : transfer learning, pre-train on low fidelity and fine-tune on high fidelity (Lejeune & Zhao 2021).
XGB      : gradient-boosted trees on log targets.
Hyper-parameters are chosen on the validation set only (see experiments)."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from .surrogates import to_s, from_s, EPS

torch.set_num_threads(1)


def _mlp(d_in, d_out, width, depth, act=nn.Tanh):
    layers = []; d = d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), act()]; d = width
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


def _train_lbfgs(params, loss_fn, iters=500):
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=iters, history_size=50, line_search_fn='strong_wolfe',
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad(); l = loss_fn(); l.backward(); return l
    opt.step(closure)


class MLP:
    def __init__(self, width=32, depth=2, wd=1e-4, n_ens=5, seed=0, bounded=False, name=None):
        self.width, self.depth, self.wd, self.n_ens, self.seed, self.bounded = width, depth, wd, n_ens, seed, bounded
        self.kind = 'logit' if bounded else 'log'
        self.name = name or ('BMLP' if bounded else 'MLP')

    def fit(self, XH, YH, LOH, UPH, **kw):
        self.sx = StandardScaler().fit(XH)
        S = to_s(YH, LOH, UPH, self.kind)
        self.sy = StandardScaler().fit(S)
        X = torch.tensor(self.sx.transform(XH), dtype=torch.float64)
        T = torch.tensor(self.sy.transform(S), dtype=torch.float64)
        self.nets = []
        for e in range(self.n_ens):
            torch.manual_seed(self.seed * 100 + e)
            net = _mlp(X.shape[1], T.shape[1], self.width, self.depth).double()
            loss = lambda: ((net(X) - T) ** 2).mean() + self.wd * sum((p ** 2).sum() for p in net.parameters())
            _train_lbfgs(net.parameters(), loss)
            self.nets.append(net)
        return self

    def predict(self, X, LO, UP, **kw):
        Xt = torch.tensor(self.sx.transform(X), dtype=torch.float64)
        with torch.no_grad():
            P = np.stack([self.sy.inverse_transform(n(Xt).numpy()) for n in self.nets])
        Y = from_s(P, LO[None], UP[None], self.kind)
        m = Y.mean(0); s = Y.std(0)
        return m, m - 1.96 * s, m + 1.96 * s


def vr_bounds(X):
    """Voigt and Reuss bounds of (G12/G_m, G23/G_m, k23/k_m) of the 2D problem (plane strain + antiplane shear) from the
    design vectors x = (V_f, lg(E_f/E_m), nu_f, nu_m, s_g); k = E / (2 (1 + nu)(1 - 2 nu)) is the plane-strain bulk modulus."""
    vf, c, nuf, num = X[:, 0], 10 ** X[:, 1], X[:, 2], X[:, 3]
    rG = c * (1 + num) / (1 + nuf)
    rk = c * (1 + num) * (1 - 2 * num) / ((1 + nuf) * (1 - 2 * nuf))
    V = lambda r: vf * r + (1 - vf)
    R = lambda r: 1 / (vf / r + (1 - vf))
    LO = np.stack([R(rG), R(rG), R(rk)], 1); UP = np.stack([V(rG), V(rG), V(rk)], 1)
    return LO, UP


class VRN:
    """Voigt-Reuss Net (Keshav & Fritzen, arXiv:2504.00712; Int. J. Numer. Methods Eng. 2026, 127: e70296), implemented
    from the paper's equations (spectral normalisation eqs. 17-22, loss eq. 25; their LGPL code,
    github.com/DataAnalyticsEngineering/VoigtReussNet, targets full tensors with image descriptors). The effective in-plane and antiplane tensors share their eigenprojectors with the Voigt
    and Reuss bounds of the 2D problem, so the spectral normalisation reduces to xi = (V - y) / (V - R) per modulus
    (multiplicities: G12 2, G23 2, k23 1). Sigmoid output = xi; loss = normalised Frobenius norm of the normalised tensor
    error (their eq. 44), averaged over samples, plus weight decay. Training as for the other networks (tanh, L-BFGS, 5 nets)."""
    W = np.array([2.0, 2.0, 1.0]) / 5.0

    def __init__(self, width=32, depth=2, wd=1e-4, n_ens=5, seed=0):
        self.width, self.depth, self.wd, self.n_ens, self.seed = width, depth, wd, n_ens, seed
        self.name = 'VRN'

    def fit(self, XH, YH, LOH, UPH, **kw):
        R, V = vr_bounds(XH)
        xi = np.clip((V - YH) / (V - R), 1e-9, 1 - 1e-9)
        self.sx = StandardScaler().fit(XH)
        X = torch.tensor(self.sx.transform(XH), dtype=torch.float64)
        T = torch.tensor(xi, dtype=torch.float64); w = torch.tensor(self.W, dtype=torch.float64)
        self.nets = []
        for e in range(self.n_ens):
            torch.manual_seed(self.seed * 100 + e)
            net = _mlp(X.shape[1], T.shape[1], self.width, self.depth).double()
            loss = lambda: torch.sqrt(((torch.sigmoid(net(X)) - T) ** 2 * w).sum(1) + 1e-12).mean() \
                + self.wd * sum((p ** 2).sum() for p in net.parameters())
            _train_lbfgs(net.parameters(), loss)
            self.nets.append(net)
        return self

    def predict(self, X, LO, UP, **kw):
        R, V = vr_bounds(X)
        Xt = torch.tensor(self.sx.transform(X), dtype=torch.float64)
        with torch.no_grad():
            xi = np.stack([torch.sigmoid(n(Xt)).numpy() for n in self.nets])
        Y = V[None] - xi * (V - R)[None]
        m = Y.mean(0); s = Y.std(0)
        return m, m - 1.96 * s, m + 1.96 * s


class TLNN(MLP):
    """Pre-train on all low-fidelity data, then fine-tune all weights on high-fidelity data."""
    def __init__(self, **kw):
        super().__init__(**kw); self.name = 'TLNN'

    def fit(self, XH, YH, LOH, UPH, XL=None, YL=None, LOL=None, UPL=None, **kw):
        self.sx = StandardScaler().fit(np.vstack([XL, XH]))
        SL, SH = to_s(YL, LOL, UPL, 'log'), to_s(YH, LOH, UPH, 'log')
        self.sy = StandardScaler().fit(SL)
        XLt = torch.tensor(self.sx.transform(XL)); XHt = torch.tensor(self.sx.transform(XH))
        TL = torch.tensor(self.sy.transform(SL)); TH = torch.tensor(self.sy.transform(SH))
        self.nets = []
        for e in range(self.n_ens):
            torch.manual_seed(self.seed * 100 + e)
            net = _mlp(XLt.shape[1], TL.shape[1], self.width, self.depth).double()
            _train_lbfgs(net.parameters(), lambda: ((net(XLt) - TL) ** 2).mean()
                         + self.wd * sum((p ** 2).sum() for p in net.parameters()))
            w0 = [p.detach().clone() for p in net.parameters()]
            _train_lbfgs(net.parameters(), lambda: ((net(XHt) - TH) ** 2).mean()
                         + self.wd * sum(((p - q) ** 2).sum() for p, q in zip(net.parameters(), w0)), iters=200)
            self.nets.append(net)
        return self


class MFNN:
    """y_L = N_L(x); y_H = N_lin(x, y_L) + N_nl(x, y_L) (Meng & Karniadakis 2020), trained jointly."""
    def __init__(self, width=20, wd=1e-4, n_ens=5, seed=0):
        self.width, self.wd, self.n_ens, self.seed = width, wd, n_ens, seed
        self.name = 'MFNN'

    def fit(self, XH, YH, LOH, UPH, XL=None, YL=None, LOL=None, UPL=None, **kw):
        self.sx = StandardScaler().fit(np.vstack([XL, XH]))
        SL, SH = to_s(YL, LOL, UPL, 'log'), to_s(YH, LOH, UPH, 'log')
        self.sy = StandardScaler().fit(SL)
        XLt = torch.tensor(self.sx.transform(XL)); XHt = torch.tensor(self.sx.transform(XH))
        TL = torch.tensor(self.sy.transform(SL)); TH = torch.tensor(self.sy.transform(SH))
        d, k = XLt.shape[1], TL.shape[1]
        self.nets = []
        for e in range(self.n_ens):
            torch.manual_seed(self.seed * 100 + e)
            NL = _mlp(d, k, self.width, 2).double()
            Nlin = nn.Linear(d + k, k).double()
            Nnl = _mlp(d + k, k, max(self.width // 2, 8), 2).double()
            params = list(NL.parameters()) + list(Nlin.parameters()) + list(Nnl.parameters())

            def loss():
                yl_h = NL(XHt)
                zH = torch.cat([XHt, yl_h], 1)
                return ((NL(XLt) - TL) ** 2).mean() + ((Nlin(zH) + Nnl(zH) - TH) ** 2).mean() \
                    + self.wd * sum((p ** 2).sum() for p in Nnl.parameters())
            _train_lbfgs(params, loss, iters=800)
            self.nets.append((NL, Nlin, Nnl))
        return self

    def predict(self, X, LO, UP, **kw):
        Xt = torch.tensor(self.sx.transform(X))
        P = []
        with torch.no_grad():
            for NL, Nlin, Nnl in self.nets:
                z = torch.cat([Xt, NL(Xt)], 1)
                P.append(self.sy.inverse_transform((Nlin(z) + Nnl(z)).numpy()))
        Y = np.exp(np.stack(P)); m = Y.mean(0); s = Y.std(0)
        return m, m - 1.96 * s, m + 1.96 * s


class XGB:
    def __init__(self, n_estimators=400, max_depth=3, learning_rate=0.05, seed=0):
        self.kw = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                       subsample=0.8, colsample_bytree=1.0, random_state=seed, n_jobs=1)
        self.name = 'XGB'

    def fit(self, XH, YH, LOH, UPH, **kw):
        import xgboost as xgb
        S = np.log(YH)
        self.models = [xgb.XGBRegressor(**self.kw).fit(XH, S[:, j]) for j in range(S.shape[1])]
        return self

    def predict(self, X, LO, UP, **kw):
        m = np.exp(np.stack([g.predict(X) for g in self.models], 1))
        return m, m, m
