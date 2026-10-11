"""
CEC2022-style benchmark suite (12 functions, F1-F12).

Notes
-----
The official CEC2022 technical report defines F1-F12 using fixed shift
vectors, rotation matrices and (for hybrid/composition functions) variable
shuffle indices that are distributed as binary data files together with the
competition code. Those data files are not available offline in this
environment, so this module reproduces the *mathematical structure* of the
CEC2022 suite (base functions, hybridisation scheme, composition scheme,
search range [-100, 100]^D and bias values) while generating the shift
vectors / rotation matrices / shuffle indices itself, seeded for
reproducibility (seed = 2022 + function_id).

This keeps the benchmark properties that matter for comparing algorithms
(shifted optima, rotated/ill-conditioned landscapes, multimodality, hybrid
and composition structure) while remaining fully self-contained.

All functions are vectorised: x has shape (..., D) and the output has shape
x.shape[:-1].
"""

from __future__ import annotations

import functools

import numpy as np

LOWER_BOUND = -100.0
UPPER_BOUND = 100.0

# Bias (offset) values used by the official CEC2022 suite.
_BIAS = {
    1: 300.0, 2: 400.0, 3: 600.0, 4: 800.0, 5: 900.0,
    6: 1800.0, 7: 2000.0, 8: 2200.0, 9: 2300.0, 10: 2400.0,
    11: 2600.0, 12: 2700.0,
}

_NAMES = {
    1: "Shifted and Rotated Zakharov",
    2: "Shifted and Rotated Rosenbrock",
    3: "Shifted and Rotated Expanded Schaffer F6",
    4: "Shifted and Rotated Non-Continuous Rastrigin",
    5: "Shifted and Rotated Levy",
    6: "Hybrid Function 1 (Bent Cigar / HGBat / Rastrigin)",
    7: "Hybrid Function 2 (Katsuura / Ackley / Schaffer F6 / Rastrigin / Schwefel)",
    8: "Hybrid Function 3 (Schwefel / Schaffer F6 / Bent Cigar / Rastrigin / Levy)",
    9: "Composition Function 1",
    10: "Composition Function 2",
    11: "Composition Function 3",
    12: "Composition Function 4",
}


# --------------------------------------------------------------------------
# Random rotation matrix / shift vector generation (deterministic per fid/dim)
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _rotation_matrix(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, d))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))
    return q


@functools.lru_cache(maxsize=None)
def _shift_vector(d: int, seed: int, scale: float = 80.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=d)


@functools.lru_cache(maxsize=None)
def _shuffle_indices(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.arange(d)
    rng.shuffle(idx)
    return idx


# --------------------------------------------------------------------------
# Base functions (operate on shifted/rotated coordinates z, shape (..., d))
# --------------------------------------------------------------------------

def _sphere(z):
    return np.sum(z ** 2, axis=-1)


def _zakharov(z):
    d = z.shape[-1]
    i = np.arange(1, d + 1)
    s = np.sum(0.5 * i * z, axis=-1)
    return np.sum(z ** 2, axis=-1) + s ** 2 + s ** 4


def _rosenbrock(z):
    zz = z * 2.048 / 100.0 + 1.0
    return np.sum(100.0 * (zz[..., :-1] ** 2 - zz[..., 1:]) ** 2
                   + (zz[..., :-1] - 1.0) ** 2, axis=-1)


def _schaffer_f6(z):
    z1 = z
    z2 = np.roll(z, -1, axis=-1)
    num = np.sin(np.sqrt(z1 ** 2 + z2 ** 2)) ** 2 - 0.5
    den = (1.0 + 0.001 * (z1 ** 2 + z2 ** 2)) ** 2
    return np.sum(0.5 + num / den, axis=-1)


def _rastrigin(z):
    zz = z * 5.12 / 100.0
    return np.sum(zz ** 2 - 10.0 * np.cos(2.0 * np.pi * zz) + 10.0, axis=-1)


def _non_cont_rastrigin(z):
    zz = z.copy()
    mask = np.abs(zz) > 0.5
    zz = np.where(mask, np.round(2.0 * zz) / 2.0, zz)
    return _rastrigin(zz)


def _levy(z):
    zz = z * 5.12 / 100.0
    w = 1.0 + (zz - 1.0) / 4.0
    term1 = np.sin(np.pi * w[..., 0]) ** 2
    term3 = (w[..., -1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[..., -1]) ** 2)
    term2 = np.sum((w[..., :-1] - 1.0) ** 2
                    * (1.0 + 10.0 * np.sin(np.pi * w[..., :-1] + 1.0) ** 2), axis=-1)
    return term1 + term2 + term3


def _bent_cigar(z):
    return z[..., 0] ** 2 + 1e6 * np.sum(z[..., 1:] ** 2, axis=-1)


def _hgbat(z):
    d = z.shape[-1]
    zz = z * 5.0 / 100.0
    s = np.sum(zz ** 2, axis=-1)
    s2 = np.sum(zz, axis=-1)
    return np.sqrt(np.abs(s ** 2 - s2 ** 2)) + (0.5 * s + s2) / d + 0.5


def _katsuura(z):
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


def _ackley(z):
    d = z.shape[-1]
    s1 = np.sum(z ** 2, axis=-1)
    s2 = np.sum(np.cos(2.0 * np.pi * z), axis=-1)
    return -20.0 * np.exp(-0.2 * np.sqrt(s1 / d)) - np.exp(s2 / d) + 20.0 + np.e


def _schwefel(z):
    d = z.shape[-1]
    zz = z * 10.0 / 100.0 + 4.209687462275036e+002
    out = np.zeros(z.shape[:-1])
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
    out = 418.9829 * d - np.sum(g, axis=-1)
    return out


def _griewank(z):
    zz = z * 6.0
    d = z.shape[-1]
    i = np.sqrt(np.arange(1, d + 1))
    return np.sum(zz ** 2, axis=-1) / 4000.0 - np.prod(np.cos(zz / i), axis=-1) + 1.0


def _discus(z):
    return 1e6 * z[..., 0] ** 2 + np.sum(z[..., 1:] ** 2, axis=-1)


def _happycat(z):
    d = z.shape[-1]
    zz = z * 5.0 / 100.0
    s = np.sum(zz ** 2, axis=-1)
    s2 = np.sum(zz, axis=-1)
    return np.abs(s - d) ** 0.25 + (0.5 * s + s2) / d + 0.5


# --------------------------------------------------------------------------
# Generic shift+rotate transform
# --------------------------------------------------------------------------

def _transform(x, shift, rot, scale=1.0):
    z = (x - shift) * scale
    return z @ rot.T


# --------------------------------------------------------------------------
# Hybrid functions: split dims into N groups (shuffled), evaluate each group
# with a different base function, sum.
# --------------------------------------------------------------------------

def _hybrid(x, fid, base_funcs, ratios):
    d = x.shape[-1]
    rot = _rotation_matrix(d, seed=2022 + fid)
    shift = _shift_vector(d, seed=3000 + fid)
    shuffle = _shuffle_indices(d, seed=4000 + fid)
    z = _transform(x, shift, rot)
    z = z[..., shuffle]

    sizes = [max(1, int(round(r * d))) for r in ratios[:-1]]
    sizes.append(d - sum(sizes))
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
# its own shift/rotation, sigma (basin width) and lambda (scale) and bias.
# --------------------------------------------------------------------------

def _composition(x, fid, base_funcs, sigmas, lambdas, biases):
    d = x.shape[-1]
    n = len(base_funcs)
    fitnesses = np.zeros(x.shape[:-1] + (n,))
    weights = np.zeros(x.shape[:-1] + (n,))

    for i, (func, sigma, lam, bias) in enumerate(zip(base_funcs, sigmas, lambdas, biases)):
        shift = _shift_vector(d, seed=5000 + fid * 100 + i, scale=80.0)
        rot = _rotation_matrix(d, seed=6000 + fid * 100 + i)
        diff = x - shift
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        w = np.where(
            dist == 0,
            np.inf,
            np.exp(-dist ** 2 / (2.0 * d * sigma ** 2)) / np.sqrt(dist + 1e-30),
        )
        z = _transform(x, shift, rot, scale=lam)
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


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def evaluate(fid: int, x: np.ndarray) -> np.ndarray:
    """Evaluate CEC2022 function `fid` (1..12) at points x (..., D)."""
    x = np.asarray(x, dtype=float)
    d = x.shape[-1]
    bias = _BIAS[fid]

    if fid == 1:
        rot = _rotation_matrix(d, seed=2022 + fid)
        shift = _shift_vector(d, seed=1000 + fid)
        z = _transform(x, shift, rot)
        return _zakharov(z) + bias

    if fid == 2:
        rot = _rotation_matrix(d, seed=2022 + fid)
        shift = _shift_vector(d, seed=1000 + fid)
        z = _transform(x, shift, rot)
        return _rosenbrock(z) + bias

    if fid == 3:
        rot = _rotation_matrix(d, seed=2022 + fid)
        shift = _shift_vector(d, seed=1000 + fid)
        z = _transform(x, shift, rot, scale=1.0)
        return _schaffer_f6(z) + bias

    if fid == 4:
        rot = _rotation_matrix(d, seed=2022 + fid)
        shift = _shift_vector(d, seed=1000 + fid)
        z = _transform(x, shift, rot)
        return _non_cont_rastrigin(z) + bias

    if fid == 5:
        rot = _rotation_matrix(d, seed=2022 + fid)
        shift = _shift_vector(d, seed=1000 + fid)
        z = _transform(x, shift, rot)
        return _levy(z) + bias

    if fid == 6:
        return _hybrid(x, fid, [_bent_cigar, _hgbat, _rastrigin], [0.4, 0.4, 0.2]) + bias

    if fid == 7:
        return _hybrid(
            x, fid,
            [_katsuura, _ackley, _schaffer_f6, _rastrigin, _schwefel],
            [0.1, 0.2, 0.2, 0.2, 0.3],
        ) + bias

    if fid == 8:
        return _hybrid(
            x, fid,
            [_schwefel, _schaffer_f6, _bent_cigar, _rastrigin, _levy],
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ) + bias

    if fid == 9:
        return _composition(
            x, fid,
            base_funcs=[_rosenbrock, _ackley, _schwefel, _rastrigin, _sphere],
            sigmas=[10, 20, 30, 40, 50],
            lambdas=[1.0, 1e-6, 1.0, 1.0, 1e-6],
            biases=[0, 200, 300, 100, 400],
        ) + bias

    if fid == 10:
        return _composition(
            x, fid,
            base_funcs=[_ackley, _schwefel, _griewank, _rosenbrock],
            sigmas=[20, 10, 30, 40],
            lambdas=[1.0, 1.0, 1e-3, 1.0],
            biases=[0, 200, 100, 300],
        ) + bias

    if fid == 11:
        return _composition(
            x, fid,
            base_funcs=[_griewank, _happycat, _schwefel, _rastrigin, _hgbat],
            sigmas=[10, 20, 30, 40, 50],
            lambdas=[1e-3, 1.0, 1.0, 1.0, 1.0],
            biases=[0, 200, 300, 400, 100],
        ) + bias

    if fid == 12:
        return _composition(
            x, fid,
            base_funcs=[_hgbat, _rastrigin, _schwefel, _bent_cigar, _griewank, _ackley],
            sigmas=[10, 20, 30, 40, 50, 60],
            lambdas=[1.0, 1.0, 1.0, 1e-6, 1e-3, 1.0],
            biases=[0, 100, 200, 300, 400, 500],
        ) + bias

    raise ValueError(f"Unknown CEC2022 function id: {fid}")


def get_function(fid: int):
    """Return a callable f(x) -> fitness for the given function id."""
    return lambda x: evaluate(fid, x)


def function_name(fid: int) -> str:
    return _NAMES[fid]


def bounds(d: int):
    lb = np.full(d, LOWER_BOUND)
    ub = np.full(d, UPPER_BOUND)
    return lb, ub


ALL_FUNCTION_IDS = list(range(1, 13))
