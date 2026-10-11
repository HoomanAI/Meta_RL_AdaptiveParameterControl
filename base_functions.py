"""Shared base functions and shift/rotate/hybrid/composition machinery for
the hand-rolled CEC2017 and CEC2022-style suites (official CEC binary data
files -- shift vectors, rotation matrices, shuffle indices -- are not
available offline in this environment; see problems/cec2017.py and
cec2022.py module docstrings). This module reproduces the mathematical
structure of the suites (same base landscapes, hybridisation/composition
scheme, search range, bias values) with self-generated, fixed-seed
shift/rotation/shuffle data, following the pattern already used and
validated in the prior RELEASE project (Simulation/benchmarks/cec2022.py).
"""

from __future__ import annotations

import functools

import numpy as np


@functools.lru_cache(maxsize=None)
def rotation_matrix(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, d))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))
    return q


@functools.lru_cache(maxsize=None)
def shift_vector(d: int, seed: int, scale: float = 80.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=d)


@functools.lru_cache(maxsize=None)
def shuffle_indices(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.arange(d)
    rng.shuffle(idx)
    return idx


def transform(x, shift, rot, scale=1.0):
    z = (x - shift) * scale
    return z @ rot.T


# --------------------------------------------------------------------------
# Base functions (operate on shifted/rotated coordinates z, shape (..., d))
# --------------------------------------------------------------------------

def sphere(z):
    return np.sum(z ** 2, axis=-1)


def elliptic(z):
    d = z.shape[-1]
    i = np.arange(d)
    exponent = 6.0 * i / max(d - 1, 1)
    return np.sum((10.0 ** exponent) * z ** 2, axis=-1)


def zakharov(z):
    d = z.shape[-1]
    i = np.arange(1, d + 1)
    s = np.sum(0.5 * i * z, axis=-1)
    return np.sum(z ** 2, axis=-1) + s ** 2 + s ** 4


def rosenbrock(z):
    zz = z * 2.048 / 100.0 + 1.0
    return np.sum(100.0 * (zz[..., :-1] ** 2 - zz[..., 1:]) ** 2
                  + (zz[..., :-1] - 1.0) ** 2, axis=-1)


def schaffer_f6(z):
    z1 = z
    z2 = np.roll(z, -1, axis=-1)
    num = np.sin(np.sqrt(z1 ** 2 + z2 ** 2)) ** 2 - 0.5
    den = (1.0 + 0.001 * (z1 ** 2 + z2 ** 2)) ** 2
    return np.sum(0.5 + num / den, axis=-1)


def rastrigin(z):
    zz = z * 5.12 / 100.0
    return np.sum(zz ** 2 - 10.0 * np.cos(2.0 * np.pi * zz) + 10.0, axis=-1)


def non_cont_rastrigin(z):
    zz = z.copy()
    mask = np.abs(zz) > 0.5
    zz = np.where(mask, np.round(2.0 * zz) / 2.0, zz)
    return rastrigin(zz)


def levy(z):
    zz = z * 5.12 / 100.0
    w = 1.0 + (zz - 1.0) / 4.0
    term1 = np.sin(np.pi * w[..., 0]) ** 2
    term3 = (w[..., -1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[..., -1]) ** 2)
    term2 = np.sum((w[..., :-1] - 1.0) ** 2
                   * (1.0 + 10.0 * np.sin(np.pi * w[..., :-1] + 1.0) ** 2), axis=-1)
    return term1 + term2 + term3


def bent_cigar(z):
    return z[..., 0] ** 2 + 1e6 * np.sum(z[..., 1:] ** 2, axis=-1)


def discus(z):
    return 1e6 * z[..., 0] ** 2 + np.sum(z[..., 1:] ** 2, axis=-1)


def hgbat(z):
    d = z.shape[-1]
    zz = z * 5.0 / 100.0
    s = np.sum(zz ** 2, axis=-1)
    s2 = np.sum(zz, axis=-1)
    return np.sqrt(np.abs(s ** 2 - s2 ** 2)) + (0.5 * s + s2) / d + 0.5


def happycat(z):
    d = z.shape[-1]
    zz = z * 5.0 / 100.0
    s = np.sum(zz ** 2, axis=-1)
    s2 = np.sum(zz, axis=-1)
    return np.abs(s - d) ** 0.25 + (0.5 * s + s2) / d + 0.5


def katsuura(z):
    d = z.shape[-1]
    zz = z * 5.0 / 100.0
    k = np.arange(1, 33)
    pow2 = (2.0 ** k)[None, None, :]
    zj = zz[..., :, None]
    term = np.abs(pow2 * zj - np.round(pow2 * zj)) / pow2
    s = np.sum(term, axis=-1)
    j = np.arange(1, d + 1)
    factor = (1.0 + j * s) ** (10.0 / d ** 1.2)
    out = np.prod(factor, axis=-1)
    return (10.0 / d ** 2) * out - (10.0 / d ** 2)


def ackley(z):
    d = z.shape[-1]
    s1 = np.sum(z ** 2, axis=-1)
    s2 = np.sum(np.cos(2.0 * np.pi * z), axis=-1)
    return -20.0 * np.exp(-0.2 * np.sqrt(s1 / d)) - np.exp(s2 / d) + 20.0 + np.e


def schwefel(z):
    d = z.shape[-1]
    zz = z * 10.0 / 100.0 + 4.209687462275036e+002
    over = zz > 500
    under = zz < -500
    in_range = ~over & ~under
    g = np.zeros_like(zz)
    g = np.where(in_range, zz * np.sin(np.sqrt(np.abs(zz))), g)
    mod_over = 500 - np.mod(zz, 500)
    g = np.where(over, mod_over * np.sin(np.sqrt(np.abs(mod_over)))
                 - (zz - 500) ** 2 / (10000.0 * d), g)
    mod_under = np.mod(np.abs(zz), 500) - 500
    g = np.where(under, mod_under * np.sin(np.sqrt(np.abs(mod_under)))
                 - (zz + 500) ** 2 / (10000.0 * d), g)
    return 418.9829 * d - np.sum(g, axis=-1)


def griewank(z):
    zz = z * 6.0
    d = z.shape[-1]
    i = np.sqrt(np.arange(1, d + 1))
    return np.sum(zz ** 2, axis=-1) / 4000.0 - np.prod(np.cos(zz / i), axis=-1) + 1.0


# --------------------------------------------------------------------------
# Hybrid functions: split (shuffled) dims into groups, one base function per
# group, sum the pieces.
# --------------------------------------------------------------------------

def hybrid(x, fid_seed, base_funcs, ratios):
    d = x.shape[-1]
    rot = rotation_matrix(d, seed=2000 + fid_seed)
    shift = shift_vector(d, seed=3000 + fid_seed)
    shuffle = shuffle_indices(d, seed=4000 + fid_seed)
    z = transform(x, shift, rot)
    z = z[..., shuffle]

    sizes = [max(1, int(round(r * d))) for r in ratios[:-1]]
    sizes.append(max(0, d - sum(sizes)))
    out = np.zeros(x.shape[:-1])
    start = 0
    for func, size in zip(base_funcs, sizes):
        if size <= 0:
            continue
        chunk = z[..., start:start + size]
        out = out + func(chunk)
        start += size
    return out


# --------------------------------------------------------------------------
# Composition functions: weighted combination of base functions, each with
# its own shift/rotation/sigma/lambda/bias.
# --------------------------------------------------------------------------

def composition(x, fid_seed, base_funcs, sigmas, lambdas, biases):
    d = x.shape[-1]
    n = len(base_funcs)
    fitnesses = np.zeros(x.shape[:-1] + (n,))
    weights = np.zeros(x.shape[:-1] + (n,))

    for i, (func, sigma, lam, bias) in enumerate(zip(base_funcs, sigmas, lambdas, biases)):
        shift = shift_vector(d, seed=5000 + fid_seed * 100 + i, scale=80.0)
        rot = rotation_matrix(d, seed=6000 + fid_seed * 100 + i)
        diff = x - shift
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        w = np.where(
            dist == 0,
            np.inf,
            np.exp(-dist ** 2 / (2.0 * d * sigma ** 2)) / np.sqrt(dist + 1e-30),
        )
        z = transform(x, shift, rot, scale=lam)
        f = func(z) + bias
        fitnesses[..., i] = f
        weights[..., i] = w

    inf_mask = np.isinf(weights)
    any_inf = np.any(inf_mask, axis=-1, keepdims=True)
    safe_weights = np.where(inf_mask, 1.0, 0.0)
    weights = np.where(any_inf, safe_weights, weights)

    wsum = np.sum(weights, axis=-1, keepdims=True)
    wsum = np.where(wsum == 0, 1e-30, wsum)
    weights = weights / wsum

    return np.sum(weights * fitnesses, axis=-1)
