"""Analytical micromechanics of unidirectional two-phase fibre composites (isotropic phases).

k   : plane-strain (transverse) bulk modulus, k = lambda + mu = E / (2 (1 + nu) (1 - 2 nu))
G12 : longitudinal shear modulus, G23 : transverse shear modulus.
References: Hashin & Rosen (1964) J. Appl. Mech. 31; Hill (1964) J. Mech. Phys. Solids 12;
Hashin (1965) J. Mech. Phys. Solids 13; Hashin (1983) J. Appl. Mech. 50; Halpin & Kardos (1976) Polym. Eng. Sci. 16.
"""
import numpy as np


def phase(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))
    return dict(E=E, nu=nu, lam=lam, mu=mu, k=lam + mu)


def hs_bounds(vf, f, m):
    """Hashin-Shtrikman bounds for (k23, G23, G12); valid for well-ordered phases (k_f >= k_m, mu_f >= mu_m)."""
    cf, cm = vf, 1 - vf
    kf, km, Gf, Gm = f['k'], m['k'], f['mu'], m['mu']
    k_lo = km + cf / (1 / (kf - km) + cm / (km + Gm))
    k_up = kf + cm / (1 / (km - kf) + cf / (kf + Gf))
    G23_lo = Gm + cf / (1 / (Gf - Gm) + cm * (km + 2 * Gm) / (2 * Gm * (km + Gm)))
    G23_up = Gf + cm / (1 / (Gm - Gf) + cf * (kf + 2 * Gf) / (2 * Gf * (kf + Gf)))
    G12_lo = Gm + cf / (1 / (Gf - Gm) + cm / (2 * Gm))
    G12_up = Gf + cm / (1 / (Gm - Gf) + cf / (2 * Gf))
    return dict(k23=(k_lo, k_up), G23=(G23_lo, G23_up), G12=(G12_lo, G12_up))


def halpin_tsai(Pf, Pm, vf, xi):
    eta = (Pf / Pm - 1) / (Pf / Pm + xi)
    return Pm * (1 + xi * eta * vf) / (1 - eta * vf)


def hill_longitudinal(vf, f, m, k23):
    """Hill's exact connections (uniform-field argument): E1 and nu12 follow from k23 for two isotropic phases,
    E1 = <E> + 4 a^2 d,  nu12 = <nu> - a d,  a = (nu_f - nu_m)/(1/k_f - 1/k_m),  d = <1/k> - 1/k23.
    (Versions before 2026-10-17 16:03 had '+ a d' for nu12, a sign error found in the pre-submission reference check;
    E1 was correct. Verified against the composite-cylinder closed form in tests/test_models.py.)"""
    cf, cm = vf, 1 - vf
    a = (f['nu'] - m['nu']) / (1 / f['k'] - 1 / m['k'])
    d = cf / f['k'] + cm / m['k'] - 1 / k23
    E1 = cf * f['E'] + cm * m['E'] + 4 * a ** 2 * d
    nu12 = cf * f['nu'] + cm * m['nu'] - a * d
    return E1, nu12


def engineering(k23, G23, G12, E1, nu12):
    """Transverse Young's modulus E2 and Poisson ratio nu23 of a transversely isotropic material."""
    mfac = 1 + 4 * k23 * nu12 ** 2 / E1
    E2 = 4 * k23 * G23 / (k23 + mfac * G23)
    nu23 = (k23 - mfac * G23) / (k23 + mfac * G23)
    return E2, nu23


def all_constants(vf, f, m, k23, G23, G12):
    E1, nu12 = hill_longitudinal(vf, f, m, k23)
    E2, nu23 = engineering(k23, G23, G12, E1, nu12)
    return dict(k23=k23, G23=G23, G12=G12, E1=E1, nu12=nu12, E2=E2, nu23=nu23)


def _iso_mandel(p):
    """Isotropic stiffness in Mandel notation (11, 22, 33, 23, 13, 12; shear components scaled by sqrt 2)."""
    C = np.zeros((6, 6)); C[:3, :3] = p['lam']; C += 2 * p['mu'] * np.eye(6)
    return C


def _eshelby_cylinder(nu):
    """Eshelby tensor (Mandel) of an infinite circular cylinder along x3 in an isotropic matrix (Mura 1987)."""
    S = np.zeros((6, 6)); a = 8 * (1 - nu)
    S[0, 0] = S[1, 1] = (5 - 4 * nu) / a; S[0, 1] = S[1, 0] = (4 * nu - 1) / a
    S[0, 2] = S[1, 2] = nu / (2 * (1 - nu))
    S[3, 3] = S[4, 4] = 2 * 0.25; S[5, 5] = 2 * (3 - 4 * nu) / a
    return S


def mori_tanaka_eshelby(vf, f, m):
    """Mori-Tanaka estimate computed independently from the Eshelby tensor (fibres along x3, matrix as reference):
    C* = Cm + cf (Cf - Cm) A [cm I + cf A]^-1,  A = [I + S Cm^-1 (Cf - Cm)]^-1.
    Returns (k23, G23, G12) with the paper's naming (1 = fibre direction)."""
    Cf, Cm = _iso_mandel(f), _iso_mandel(m); I = np.eye(6); S = _eshelby_cylinder(m['nu'])
    A = np.linalg.inv(I + S @ np.linalg.solve(Cm, Cf - Cm))
    C = Cm + vf * (Cf - Cm) @ A @ np.linalg.inv((1 - vf) * I + vf * A)
    return dict(k23=(C[0, 0] + C[0, 1]) / 2, G23=C[5, 5] / 2, G12=C[3, 3] / 2)


def mori_tanaka(vf, f, m):
    """Mori-Tanaka estimate for matrix-based UD composites = Hashin-Shtrikman lower bound
    (identical to mori_tanaka_eshelby to round-off; see tests/test_models.py)."""
    b = hs_bounds(vf, f, m)
    return dict(k23=b['k23'][0], G23=b['G23'][0], G12=b['G12'][0])


def halpin_tsai_set(vf, f, m):
    """Halpin-Tsai estimates with the reinforcement factors of Halpin & Kardos (1976): xi = 1 for G12 and
    xi = k_m / (k_m + 2 G_m) for G23 (k_m: plane-strain bulk modulus); k23 is taken as the Hashin lower value.
    With these factors HT coincides identically with the HS lower bound (= Mori-Tanaka) for G12 and G23.
    (Versions before 2026-10-17 used xi = 1/(4 - 3 nu_m) for G23, which is not the Halpin-Kardos factor.)"""
    G12 = halpin_tsai(f['mu'], m['mu'], vf, 1.0)
    G23 = halpin_tsai(f['mu'], m['mu'], vf, m['k'] / (m['k'] + 2 * m['mu']))
    k23 = hs_bounds(vf, f, m)['k23'][0]
    return dict(k23=k23, G23=G23, G12=G12)
