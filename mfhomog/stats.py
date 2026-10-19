"""Paired statistics over repeats: Wilcoxon signed-rank test with Holm correction, effect sizes."""
import numpy as np
from scipy.stats import wilcoxon


def holm(pvals):
    p = np.asarray(pvals, float); n = len(p); order = np.argsort(p)
    adj = np.empty(n); running = 0.0
    for k, i in enumerate(order):
        running = max(running, (n - k) * p[i]); adj[i] = min(1.0, running)
    return adj


def paired(a, b):
    """a: errors of our method, b: errors of a baseline, paired by repeat.
    Returns p (two-sided Wilcoxon), wins of a, median relative reduction (b - a) / b, rank-biserial r."""
    a = np.asarray(a); b = np.asarray(b)
    d = b - a
    if np.allclose(d, 0):
        return 1.0, 0, 0.0, 0.0
    res = wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
    nz = d[d != 0]; ranks = np.argsort(np.argsort(np.abs(nz))) + 1
    rb = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
    return float(res.pvalue), int((a < b).sum()), float(np.median(d / b)), float(rb)
