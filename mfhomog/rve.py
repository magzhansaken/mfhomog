"""Periodic random fibre arrangements in the transverse plane (unit square cell, monodisperse disks).

Arrangements are equilibrated hard-disk configurations: start from a perturbed hexagonal-like lattice,
then Metropolis displacement moves under a hard-core constraint (minimum centre distance 2 r (1 + gap)).
The same continuous geometry is rasterised at any resolution, so all fidelity levels see one microstructure.
"""
import numpy as np


def _min_image(d):
    return d - np.round(d)


def generate(vf, n_fibres=36, gap=0.1, sweeps=400, seed=0, max_step=None):
    """Return fibre centres (n,2) in [0,1)^2 and radius r for volume fraction vf."""
    rng = np.random.default_rng(seed)
    r = np.sqrt(vf / (n_fibres * np.pi))
    dmin = 2 * r * (1 + gap)
    # initial lattice: rows of a slightly irregular grid that satisfies the hard-core distance
    nx = int(np.ceil(np.sqrt(n_fibres)))
    ny = int(np.ceil(n_fibres / nx))
    pts = []
    for j in range(ny):
        for i in range(nx):
            if len(pts) < n_fibres:
                pts.append([(i + 0.5 * (j % 2) + 0.25) / nx, (j + 0.5) / ny])
    X = np.array(pts) % 1.0
    if not _valid(X, dmin):
        raise ValueError('initial lattice violates hard-core distance; reduce vf or gap')
    step = max_step if max_step is not None else 0.5 * (1.0 / np.sqrt(n_fibres) - dmin)
    step = max(step, 0.05 * r)
    acc = 0; tot = 0
    for s in range(sweeps):
        for i in rng.permutation(n_fibres):
            trial = (X[i] + rng.uniform(-step, step, 2)) % 1.0
            d = _min_image(X - trial); dist = np.sqrt((d ** 2).sum(1)); dist[i] = np.inf
            tot += 1
            if dist.min() >= dmin:
                X[i] = trial; acc += 1
        if s % 20 == 19:  # adapt step towards ~40 % acceptance
            rate = acc / max(tot, 1); step *= 1.2 if rate > 0.45 else (0.8 if rate < 0.35 else 1.0)
            acc = tot = 0
    return X, r


def _valid(X, dmin):
    for i in range(len(X)):
        d = _min_image(X - X[i]); dist = np.sqrt((d ** 2).sum(1)); dist[i] = np.inf
        if dist.min() < dmin * (1 - 1e-12):
            return False
    return True


def rasterise(X, r, N):
    """Pixel-centre sampling of the periodic disk arrangement on an N x N grid (True = fibre)."""
    g = (np.arange(N) + 0.5) / N
    PX, PY = np.meshgrid(g, g, indexing='ij')
    img = np.zeros((N, N), bool)
    for c in X:
        dx = _min_image(PX - c[0]); dy = _min_image(PY - c[1])
        img |= dx ** 2 + dy ** 2 <= r ** 2
    return img
