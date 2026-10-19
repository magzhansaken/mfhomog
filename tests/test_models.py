"""Fast unit tests of the analytical relations and the surrogate interface (run: python3 -m pytest tests -q)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mfhomog import micromech as mm, surrogates as S


def test_hill_relations_homogeneous_limit():
    # identical phases: E1 = E, nu12 = nu, E2 = E, nu23 = nu
    p = mm.phase(3.0, 0.3); q = mm.phase(3.0 * (1 + 1e-9), 0.3)
    c = mm.all_constants(0.4, q, p, p['k'], p['mu'], p['mu'])
    assert abs(c['E1'] - 3.0) < 1e-6 and abs(c['nu12'] - 0.3) < 1e-6
    assert abs(c['E2'] - 3.0) < 1e-6 and abs(c['nu23'] - 0.3) < 1e-6


def test_bounds_ordered_and_mt_is_lower():
    f, m = mm.phase(70.0, 0.22), mm.phase(3.0, 0.35)
    for vf in (0.2, 0.4, 0.65):
        b = mm.hs_bounds(vf, f, m); mt = mm.mori_tanaka(vf, f, m)
        for k in ('G12', 'G23', 'k23'):
            assert b[k][0] < b[k][1]
            assert abs(mt[k] - b[k][0]) < 1e-12


def test_halpin_tsai_equals_hs_lower_bound():
    # with xi = 1 (G12) and xi = k_m/(k_m + 2 G_m) (G23) [Halpin & Kardos 1976] HT is identical to the HS lower bound
    for ef, nf, nm in ((5.0, 0.15, 0.42), (40.0, 0.22, 0.35), (100.0, 0.30, 0.30)):
        f, m = mm.phase(ef, nf), mm.phase(1.0, nm)
        for vf in (0.2, 0.45, 0.65):
            ht, mt = mm.halpin_tsai_set(vf, f, m), mm.mori_tanaka(vf, f, m)
            for k in ('G12', 'G23', 'k23'):
                assert abs(ht[k] / mt[k] - 1) < 1e-12


def test_mori_tanaka_eshelby_equals_hs_lower_bound():
    # independent Eshelby-tensor Mori-Tanaka (infinite circular cylinders) reproduces the HS lower bound
    for ef, nf, nm in ((5.0, 0.15, 0.42), (40.0, 0.22, 0.35), (100.0, 0.30, 0.30)):
        f, m = mm.phase(ef, nf), mm.phase(1.0, nm)
        for vf in (0.2, 0.45, 0.65):
            me, b = mm.mori_tanaka_eshelby(vf, f, m), mm.hs_bounds(vf, f, m)
            for k in ('G12', 'G23', 'k23'):
                assert abs(me[k] / b[k][0] - 1) < 1e-12


def test_bound_normalised_transform_roundtrip_and_range():
    rng = np.random.default_rng(0)
    LO = rng.uniform(1, 2, (50, 3)); UP = LO + rng.uniform(0.5, 3, (50, 3)); Y = LO + rng.uniform(0.01, 0.99, (50, 3)) * (UP - LO)
    s = S.to_s(Y, LO, UP, 'logit')
    assert np.allclose(S.from_s(s, LO, UP, 'logit'), Y)
    big = S.from_s(np.full_like(s, 50.0), LO, UP, 'logit'); small = S.from_s(np.full_like(s, -50.0), LO, UP, 'logit')
    assert np.all(big <= UP) and np.all(small >= LO)


def test_corrector_predictions_respect_bounds():
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(15, 5)); LO = 1 + X[:, :3]; UP = LO + 2
    YH = LO + 0.2 * (UP - LO) + 0.01 * rng.standard_normal((15, 3)); YL = YH * 1.03
    m = S.Corrector('logit', 0).fit(X, YH, LO, UP, YLH=np.minimum(YL, UP - 1e-3))
    Xq = rng.uniform(size=(20, 5)); LOq = 1 + Xq[:, :3]; UPq = LOq + 2
    mu, lo, up = m.predict(Xq, LOq, UPq, YLq=LOq + 0.25 * (UPq - LOq))
    assert np.all(lo >= LOq - 1e-12) and np.all(up <= UPq + 1e-12) and np.all((mu >= lo) & (mu <= up))


def test_hill_connections_match_composite_cylinder():
    """With k23 = HS lower bound (= composite cylinder assemblage), Hill's E1 and nu12 must equal the closed forms of
    the composite cylinder assemblage (Hashin 1983, J. Appl. Mech. 50)."""
    import itertools
    for c, nuf, num, vf in itertools.product([5.0, 20.0, 100.0], [0.15, 0.3], [0.3, 0.42], [0.2, 0.45, 0.65]):
        f, m = mm.phase(c, nuf), mm.phase(1.0, num); vm = 1 - vf
        k23 = mm.hs_bounds(vf, f, m)['k23'][0]
        E1, nu12 = mm.hill_longitudinal(vf, f, m, k23)
        den = vm / f['k'] + vf / m['k'] + 1 / m['mu']
        nu_cca = m['nu'] * vm + f['nu'] * vf + (f['nu'] - m['nu']) * (1 / m['k'] - 1 / f['k']) * vm * vf / den
        E_cca = m['E'] * vm + f['E'] * vf + 4 * (f['nu'] - m['nu']) ** 2 * vm * vf / den
        assert abs(nu12 - nu_cca) < 1e-12 and abs(E1 / E_cca - 1) < 1e-12
