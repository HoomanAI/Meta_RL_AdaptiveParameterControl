"""Shared statistics helpers: Holm correction, Vargha-Delaney A12, paired Wilcoxon."""
import numpy as np
from scipy.stats import wilcoxon


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); adj = np.empty_like(ps); m = len(ps); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, (m - k) * ps[i])); adj[i] = run
    return adj


def a12(x, y):
    """Probability that a run of x is better (smaller) than a run of y (ties count 1/2)."""
    x = np.asarray(x)[:, None]; y = np.asarray(y)[None, :]
    return float(np.mean(x < y) + 0.5 * np.mean(x == y))


def wil(x, y):
    """Two-sided paired Wilcoxon signed-rank p-value (zero differences discarded)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.all(np.abs(x - y) <= 1e-12 * (1 + np.abs(y))):
        return 1.0
    try:
        return float(wilcoxon(x, y).pvalue)
    except ValueError:
        return 1.0
