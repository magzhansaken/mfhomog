"""Load generated samples, build targets for every fidelity level and the physics quantities
(Hashin-Shtrikman bounds, analytical estimates) needed by the surrogates."""
import json
import numpy as np
from . import micromech as mm

TARGETS = ['G12', 'G23', 'k23']


def load(path):
    return [json.loads(l) for l in open(path)]


def physics(rec):
    p = rec['phys']
    f = mm.phase(p['contrast'], p['nu_f']); m = mm.phase(1.0, p['nu_m'])
    return p, f, m


def normalised(rec, level):
    """Targets normalised by the matrix moduli: G12/Gm, G23/Gm, k23/km at a given level.
    level 'HF' = Richardson extrapolation from 256 and 512 (first-order convergence in h)."""
    p, f, m = physics(rec)
    L = rec['levels']
    if level == 'HF':
        v = {t: 2 * L['512'][t] - L['256'][t] for t in TARGETS}
    else:
        v = {t: L[str(level)][t] for t in TARGETS}
    return np.array([v['G12'] / m['mu'], v['G23'] / m['mu'], v['k23'] / m['k']])


def bounds(rec):
    """HS lower/upper bounds (normalised) for G12, G23, k23 at the nominal fibre volume fraction."""
    p, f, m = physics(rec)
    b = mm.hs_bounds(p['vf'], f, m)
    lo = np.array([b['G12'][0] / m['mu'], b['G23'][0] / m['mu'], b['k23'][0] / m['k']])
    up = np.array([b['G12'][1] / m['mu'], b['G23'][1] / m['mu'], b['k23'][1] / m['k']])
    return lo, up


def analytic(rec, kind):
    p, f, m = physics(rec)
    e = mm.mori_tanaka(p['vf'], f, m) if kind == 'MT' else mm.halpin_tsai_set(p['vf'], f, m)
    return np.array([e['G12'] / m['mu'], e['G23'] / m['mu'], e['k23'] / m['k']])


def engineering_from(rec, y):
    """Derived constants (E2/Em, nu23, E1/Em, nu12) from normalised (G12, G23, k23) via Hill's exact connections."""
    p, f, m = physics(rec)
    G12, G23, k23 = y[0] * m['mu'], y[1] * m['mu'], y[2] * m['k']
    c = mm.all_constants(p['vf'], f, m, k23, G23, G12)
    return np.array([c['E2'] / m['E'], c['nu23'], c['E1'] / m['E'], c['nu12']])


def arrays(recs, level):
    X = np.array([r['x'] for r in recs])
    Y = np.array([normalised(r, level) for r in recs])
    LO = np.array([bounds(r)[0] for r in recs]); UP = np.array([bounds(r)[1] for r in recs])
    return X, Y, LO, UP


def cost(recs, level):
    """CPU seconds actually spent per sample at a level (HF = 256 + 512 solves)."""
    out = []
    for r in recs:
        L = r['levels']
        if level == 'HF':
            out.append(sum(L['256']['t']) + sum(L['512']['t']) + r['t_geo'])
        else:
            out.append(sum(L[str(level)]['t']) + r['t_geo'])
    return np.array(out)
