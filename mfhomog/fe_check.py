"""Independent cross-check solver: bilinear (Q1) finite elements on the pixel grid with periodic
fluctuation boundary conditions. Same pixel geometry as the FFT solver, different discretisation."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

_gp = np.array([-1, 1]) / np.sqrt(3.0)


def _q1_grad(h):
    """Gradients of the 4 bilinear shape functions at the 4 Gauss points of a square element of side h."""
    out = []
    for xi in _gp:
        for eta in _gp:
            dN = 0.25 * np.array([[-(1 - eta), -(1 - xi)], [(1 - eta), -(1 + xi)],
                                  [(1 + eta), (1 + xi)], [-(1 + eta), (1 - xi)]]) * (2.0 / h)
            out.append(dN)
    return out, (h / 2.0) ** 2  # weight * detJ for each Gauss point


def _connectivity(N):
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    n = lambda i, j: (i % N) * N + (j % N)
    return np.stack([n(I, J), n(I + 1, J), n(I + 1, J + 1), n(I, J + 1)], -1).reshape(-1, 4)


def antiplane_fe(phase, Gf, Gm):
    N = phase.shape[0]; h = 1.0 / N
    G = np.where(phase, Gf, Gm).ravel()
    conn = _connectivity(N)
    dNs, wdet = _q1_grad(h)
    Ke = sum(dN @ dN.T for dN in dNs) * wdet          # 4x4 reference stiffness (unit modulus)
    be = [sum(dN[:, a] for dN in dNs) * wdet for a in range(2)]  # load vectors for unit macro-gradients
    rows = np.repeat(conn, 4, 1).ravel(); cols = np.tile(conn, (1, 4)).ravel()
    K = sp.csr_matrix(((G[:, None, None] * Ke[None]).ravel(), (rows, cols)), shape=(N * N, N * N))
    Geff = np.zeros((2, 2))
    keep = np.arange(1, N * N)
    Kr = K[keep][:, keep].tocsc(); lu = spl.splu(Kr)
    for a in range(2):
        f = np.zeros(N * N)
        np.add.at(f, conn.ravel(), (-G[:, None] * be[a][None]).ravel())
        u = np.zeros(N * N); u[keep] = lu.solve(f[keep])
        # average flux
        for b in range(2):
            tot = 0.0
            for dN in dNs:
                grad = (u[conn] @ dN)[:, b] + (1.0 if a == b else 0.0)
                tot += np.sum(G * grad) * wdet
            Geff[b, a] = tot
    return Geff


def plane_strain_fe(phase, lam_f, mu_f, lam_m, mu_m):
    N = phase.shape[0]; h = 1.0 / N
    lam = np.where(phase, lam_f, lam_m).ravel(); mu = np.where(phase, mu_f, mu_m).ravel()
    conn = _connectivity(N)
    dofs = np.concatenate([2 * conn, 2 * conn + 1], 1)[:, [0, 4, 1, 5, 2, 6, 3, 7]]  # (ux,uy) per node
    dNs, wdet = _q1_grad(h)
    Bs = []
    for dN in dNs:
        B = np.zeros((3, 8))
        B[0, 0::2] = dN[:, 0]; B[1, 1::2] = dN[:, 1]; B[2, 0::2] = dN[:, 1]; B[2, 1::2] = dN[:, 0]
        Bs.append(B)
    D_lam = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], float)
    D_mu = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], float)
    K_lam = sum(B.T @ D_lam @ B for B in Bs) * wdet; K_mu = sum(B.T @ D_mu @ B for B in Bs) * wdet
    vals = lam[:, None, None] * K_lam[None] + mu[:, None, None] * K_mu[None]
    rows = np.repeat(dofs, 8, 1).ravel(); cols = np.tile(dofs, (1, 8)).ravel()
    ndof = 2 * N * N
    K = sp.csr_matrix((vals.ravel(), (rows, cols)), shape=(ndof, ndof))
    keep = np.arange(2, ndof); lu = spl.splu(K[keep][:, keep].tocsc())
    Ceff = np.zeros((3, 3))
    for a, E in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):  # engineering shear strain
        E = np.array(E, float)
        Fe_lam = -sum(B.T @ D_lam @ E for B in Bs) * wdet; Fe_mu = -sum(B.T @ D_mu @ E for B in Bs) * wdet
        f = np.zeros(ndof)
        np.add.at(f, dofs.ravel(), (lam[:, None] * Fe_lam[None] + mu[:, None] * Fe_mu[None]).ravel())
        u = np.zeros(ndof); u[keep] = lu.solve(f[keep])
        S = np.zeros(3)
        for B in Bs:
            eps = u[dofs] @ B.T + E[None]
            sig = lam[:, None] * (eps @ D_lam.T) + mu[:, None] * (eps @ D_mu.T)
            S += sig.sum(0) * wdet
        Ceff[:, a] = S
    return Ceff
