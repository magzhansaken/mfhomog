"""Input space of the study and mapping from unit-cube samples to physical parameters."""
import numpy as np
from scipy.stats import qmc

NAMES = ['vf', 'log10_contrast', 'nu_f', 'nu_m', 's_gap']
LO = np.array([0.20, np.log10(5.0), 0.15, 0.30, 0.0])
HI = np.array([0.65, np.log10(100.0), 0.30, 0.42, 1.0])
PHI_MAX = 0.69   # cap on vf (1 + gap)^2 to keep the hard-disk arrangement random (below the freezing point)
G_MIN, G_CAP = 0.02, 0.30


def gap_from(vf, s):
    gmax = min(G_CAP, np.sqrt(PHI_MAX / vf) - 1.0)
    return G_MIN + s * (gmax - G_MIN)


def sobol(n, seed):
    m = int(np.ceil(np.log2(max(n, 2))))
    u = qmc.Sobol(d=len(NAMES), scramble=True, seed=seed).random(2 ** m)[:n]
    return LO + u * (HI - LO)


def to_phys(x):
    vf, lc, nuf, num, s = x
    return dict(vf=float(vf), contrast=float(10 ** lc), nu_f=float(nuf), nu_m=float(num),
                gap=float(gap_from(vf, s)))
