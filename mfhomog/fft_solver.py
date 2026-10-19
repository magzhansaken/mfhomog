"""FFT-Galerkin homogenization of 2D periodic two-phase RVEs (unidirectional fibre composites).

Two cell problems on the transverse plane of a fibre composite:
  * anti-plane (longitudinal) shear: scalar field, gives the longitudinal shear modulus G12;
  * plane strain: vector field, gives the transverse plane-strain bulk modulus k23 and shear modulus G23.

Method: Galerkin formulation with the reference-independent orthogonal projection onto compatible
fields (Zeman et al. 2010, J. Comput. Phys. 229; Vondrejc, Zeman, Marek 2014, Comput. Math. Appl. 68),
solved by conjugate gradients. Phase properties are assigned per pixel (pixel-centre sampling).
"""
import numpy as np


def _freqs(N):
    kx = np.fft.fftfreq(N); ky = np.fft.rfftfreq(N)  # real-to-complex transform along the last axis
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1.0
    nx, ny = KX / np.sqrt(K2), KY / np.sqrt(K2)
    nx[0, 0] = 0.0; ny[0, 0] = 0.0
    return nx, ny


def _cg(apply, b, tol, maxiter):
    """Conjugate gradients for a symmetric positive (semi)definite operator on the compatible subspace."""
    x = np.zeros_like(b)
    r = b.copy(); p = r.copy(); rr = np.vdot(r, r).real
    b2 = np.sqrt(np.vdot(b, b).real) or 1.0
    it = 0
    while np.sqrt(rr) / b2 > tol and it < maxiter:
        Ap = apply(p)
        alpha = rr / np.vdot(p, Ap).real
        x += alpha * p; r -= alpha * Ap
        rr_new = np.vdot(r, r).real
        p = r + (rr_new / rr) * p; rr = rr_new; it += 1
    return x, it, np.sqrt(rr) / b2


def antiplane_shear(phase, Gf, Gm, tol=1e-7, maxiter=2000, return_fields=False):
    """Effective 2x2 longitudinal shear tensor of a periodic RVE.
    phase: (N,N) bool array, True = fibre. Returns (Geff 2x2, iterations, residual)."""
    N = phase.shape[0]
    G = np.where(phase, Gf, Gm).astype(float)
    nx, ny = _freqs(N)

    def P(tx, ty):  # projection of a vector field onto gradients (curl-free, zero mean)
        fx, fy = np.fft.rfft2(tx), np.fft.rfft2(ty)
        s = nx * fx + ny * fy
        return np.fft.irfft2(nx * s, s=tx.shape), np.fft.irfft2(ny * s, s=tx.shape)

    def A(v):
        e = v.reshape(2, N, N)
        gx, gy = P(G * e[0], G * e[1])
        return np.concatenate([gx.ravel(), gy.ravel()])

    Geff = np.zeros((2, 2)); its = []; ress = []; fields = []
    for j, E in enumerate(([1.0, 0.0], [0.0, 1.0])):
        bx, by = P(G * E[0], G * E[1])
        x, it, res = _cg(A, -np.concatenate([bx.ravel(), by.ravel()]), tol, maxiter)
        e = x.reshape(2, N, N); ex = e[0] + E[0]; ey = e[1] + E[1]
        Geff[0, j] = np.mean(G * ex); Geff[1, j] = np.mean(G * ey)
        its.append(it); ress.append(res); fields.append((G * ex, G * ey))   # shear stresses (sigma_x3, sigma_y3)
    if return_fields:
        return Geff, max(its), max(ress), fields
    return Geff, max(its), max(ress)


def plane_strain(phase, lam_f, mu_f, lam_m, mu_m, tol=1e-7, maxiter=3000, return_fields=False):
    """Effective in-plane stiffness (Voigt 3x3, engineering shear strain) of a periodic RVE in plane strain.
    Tensor components are carried as full symmetric 2x2 fields (e11, e22, e12)."""
    N = phase.shape[0]
    lam = np.where(phase, lam_f, lam_m).astype(float)
    mu = np.where(phase, mu_f, mu_m).astype(float)
    nx, ny = _freqs(N)

    def C(e11, e22, e12):
        tr = e11 + e22
        return lam * tr + 2 * mu * e11, lam * tr + 2 * mu * e22, 2 * mu * e12

    def P(t11, t22, t12):
        # orthogonal (Frobenius) projection onto compatible symmetric gradients:
        # P(tau) = n (x) (tau n) + (tau n) (x) n - (n.tau.n) n (x) n
        f11, f22, f12 = np.fft.rfft2(t11), np.fft.rfft2(t22), np.fft.rfft2(t12)
        a1 = f11 * nx + f12 * ny       # (tau n)_1
        a2 = f12 * nx + f22 * ny       # (tau n)_2
        q = nx * a1 + ny * a2          # n.tau.n
        p11 = 2 * nx * a1 - q * nx * nx
        p22 = 2 * ny * a2 - q * ny * ny
        p12 = nx * a2 + ny * a1 - q * nx * ny
        sh = t11.shape
        return (np.fft.irfft2(p11, s=sh), np.fft.irfft2(p22, s=sh), np.fft.irfft2(p12, s=sh))

    def A(v):
        e = v.reshape(3, N, N)
        s = C(e[0], e[1], e[2])
        p = P(*s)
        return np.concatenate([p[0].ravel(), p[1].ravel(), p[2].ravel()])

    Ceff = np.zeros((3, 3)); its = []; ress = []; fields = []
    loads = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5])  # last: engineering shear gamma12 = 1
    for j, E in enumerate(loads):
        ones = np.ones((N, N))
        s0 = C(E[0] * ones, E[1] * ones, E[2] * ones)
        b = P(*s0)
        rhs = -np.concatenate([b[0].ravel(), b[1].ravel(), b[2].ravel()])
        x, it, res = _cg_frob(A, rhs, tol, maxiter, N)
        e = x.reshape(3, N, N)
        s = C(e[0] + E[0], e[1] + E[1], e[2] + E[2])
        Ceff[:, j] = [np.mean(s[0]), np.mean(s[1]), np.mean(s[2])]
        its.append(it); ress.append(res); fields.append(s)                 # stresses (s11, s22, s12) per load
    if return_fields:
        return Ceff, max(its), max(ress), fields
    return Ceff, max(its), max(ress)


def _cg_frob(apply, b, tol, maxiter, N):
    """CG in the Frobenius inner product for (e11, e22, e12) fields: <a,b> = a11 b11 + a22 b22 + 2 a12 b12."""
    w = np.ones(3 * N * N); w[2 * N * N:] = 2.0
    dot = lambda u, v: float(np.dot(w * u, v))
    x = np.zeros_like(b); r = b.copy(); p = r.copy(); rr = dot(r, r)
    b2 = np.sqrt(dot(b, b)) or 1.0
    it = 0
    while np.sqrt(rr) / b2 > tol and it < maxiter:
        Ap = apply(p)
        alpha = rr / dot(p, Ap)
        x += alpha * p; r -= alpha * Ap
        rr_new = dot(r, r)
        p = r + (rr_new / rr) * p; rr = rr_new; it += 1
    return x, it, np.sqrt(rr) / b2


def transverse_moduli(Ceff):
    """Isotropic projection of the in-plane effective stiffness: plane-strain bulk modulus k23 and shear G23,
    plus an anisotropy indicator (relative deviation of the RVE from transverse isotropy)."""
    C11, C22, C12, C66 = Ceff[0, 0], Ceff[1, 1], 0.5 * (Ceff[0, 1] + Ceff[1, 0]), Ceff[2, 2]
    k23 = (C11 + C22 + 2 * C12) / 4.0
    G_a = (C11 + C22 - 2 * C12) / 4.0
    G23 = 0.5 * (G_a + C66)
    aniso = abs(G_a - C66) / G23 + abs(C11 - C22) / (C11 + C22)
    return k23, G23, aniso
