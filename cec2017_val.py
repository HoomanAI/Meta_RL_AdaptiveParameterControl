"""CEC2017-style benchmark suite (30 functions, F1-F30), hand-rolled in the
same spirit and for the same reason as problems/cec2022.py: the official
CEC2017 shift/rotation/shuffle binary data files are not available offline.
This module reproduces the suite's *structure* -- Table A1's four
categories (F1-3 unimodal, F4-10 basic multimodal, F11-20 hybrid combining
sub-functions over randomly partitioned dimension subsets, F21-30
composition as weighted mixtures of sub-functions), search range
[-100,100]^D, and per-function bias -- using self-generated, fixed-seed
shift/rotation/shuffle data (problems/_base_functions.py). Per-function
optimum locations will not match the official CEC2017 ones bit-for-bit;
treat this as a structurally faithful synthetic suite, not a drop-in
replacement for the official benchmark data.
"""

from __future__ import annotations

import numpy as np

import base_functions as bf

LOWER_BOUND = -100.0
UPPER_BOUND = 100.0

CATEGORY = {}
for _fid in range(1, 4):
    CATEGORY[_fid] = "unimodal"
for _fid in range(4, 11):
    CATEGORY[_fid] = "basic_multimodal"
for _fid in range(11, 21):
    CATEGORY[_fid] = "hybrid"
for _fid in range(21, 31):
    CATEGORY[_fid] = "composition"

_BIAS = {fid: 100.0 * fid for fid in range(1, 31)}

_NAMES = {
    1: "Shifted/Rotated High Conditioned Elliptic",
    2: "Shifted/Rotated Bent Cigar",
    3: "Shifted/Rotated Zakharov",
    4: "Shifted/Rotated Rosenbrock",
    5: "Shifted/Rotated Rastrigin",
    6: "Shifted/Rotated Expanded Schaffer F6",
    7: "Shifted/Rotated Non-Continuous Rastrigin",
    8: "Shifted/Rotated Ackley",
    9: "Shifted/Rotated Griewank",
    10: "Shifted/Rotated Schwefel",
}
for _i in range(11, 21):
    _NAMES[_i] = f"Hybrid Function {_i - 10}"
for _i in range(21, 31):
    _NAMES[_i] = f"Composition Function {_i - 20}"

_UNIMODAL = {1: bf.elliptic, 2: bf.bent_cigar, 3: bf.zakharov}
_BASIC_MULTIMODAL = {
    4: bf.rosenbrock, 5: bf.rastrigin, 6: bf.schaffer_f6, 7: bf.non_cont_rastrigin,
    8: bf.ackley, 9: bf.griewank, 10: bf.schwefel,
}

# Hybrid functions F11-F20: 10 combinations of 3-6 base functions each,
# mirroring the official suite's progression of increasing component count.
_HYBRID_SPECS = {
    11: ([bf.zakharov, bf.rosenbrock, bf.rastrigin], [0.3, 0.3, 0.4]),
    12: ([bf.elliptic, bf.schwefel, bf.bent_cigar], [0.3, 0.4, 0.3]),
    13: ([bf.bent_cigar, bf.hgbat, bf.rastrigin], [0.3, 0.3, 0.4]),
    14: ([bf.hgbat, bf.rastrigin, bf.schwefel, bf.griewank], [0.25, 0.25, 0.25, 0.25]),
    15: ([bf.katsuura, bf.happycat, bf.griewank, bf.rosenbrock], [0.2, 0.3, 0.2, 0.3]),
    16: ([bf.schwefel, bf.ackley, bf.rastrigin, bf.discus], [0.25, 0.25, 0.25, 0.25]),
    17: ([bf.katsuura, bf.ackley, bf.schaffer_f6, bf.rastrigin, bf.schwefel],
         [0.1, 0.2, 0.2, 0.2, 0.3]),
    18: ([bf.elliptic, bf.ackley, bf.rastrigin, bf.hgbat, bf.discus],
         [0.2, 0.2, 0.2, 0.2, 0.2]),
    19: ([bf.bent_cigar, bf.schwefel, bf.schaffer_f6, bf.rastrigin, bf.griewank, bf.happycat],
         [0.15, 0.15, 0.2, 0.2, 0.15, 0.15]),
    20: ([bf.happycat, bf.katsuura, bf.ackley, bf.rastrigin, bf.schwefel, bf.schaffer_f6],
         [0.15, 0.15, 0.2, 0.2, 0.15, 0.15]),
}

# Composition functions F21-F30: 10 weighted mixtures of 3-6 sub-functions.
_COMPOSITION_SPECS = {
    21: dict(base_funcs=[bf.rosenbrock, bf.ackley, bf.rastrigin],
             sigmas=[10, 20, 30], lambdas=[1.0, 1e-6, 1.0], biases=[0, 100, 200]),
    22: dict(base_funcs=[bf.rastrigin, bf.griewank, bf.schwefel],
             sigmas=[10, 20, 30], lambdas=[1.0, 1e-3, 1.0], biases=[0, 100, 200]),
    23: dict(base_funcs=[bf.rosenbrock, bf.ackley, bf.schwefel, bf.rastrigin],
             sigmas=[10, 20, 30, 40], lambdas=[1.0, 1e-6, 1.0, 1.0], biases=[0, 100, 200, 300]),
    24: dict(base_funcs=[bf.ackley, bf.schwefel, bf.griewank, bf.rosenbrock],
             sigmas=[20, 10, 30, 40], lambdas=[1.0, 1.0, 1e-3, 1.0], biases=[0, 100, 200, 300]),
    25: dict(base_funcs=[bf.griewank, bf.happycat, bf.schwefel, bf.rastrigin],
             sigmas=[10, 20, 30, 40], lambdas=[1e-3, 1.0, 1.0, 1.0], biases=[0, 100, 200, 300]),
    26: dict(base_funcs=[bf.hgbat, bf.rastrigin, bf.schwefel, bf.bent_cigar, bf.griewank],
             sigmas=[10, 20, 30, 40, 50], lambdas=[1.0, 1.0, 1.0, 1e-6, 1e-3],
             biases=[0, 100, 200, 300, 400]),
    27: dict(base_funcs=[bf.rosenbrock, bf.ackley, bf.schwefel, bf.rastrigin, bf.sphere],
             sigmas=[10, 20, 30, 40, 50], lambdas=[1.0, 1e-6, 1.0, 1.0, 1e-6],
             biases=[0, 100, 200, 300, 400]),
    28: dict(base_funcs=[bf.katsuura, bf.happycat, bf.griewank, bf.rosenbrock, bf.discus],
             sigmas=[10, 20, 30, 40, 50], lambdas=[1.0, 1.0, 1e-3, 1.0, 1e-6],
             biases=[0, 100, 200, 300, 400]),
    29: dict(base_funcs=[bf.hgbat, bf.rastrigin, bf.schwefel, bf.bent_cigar, bf.griewank, bf.ackley],
             sigmas=[10, 20, 30, 40, 50, 60], lambdas=[1.0, 1.0, 1.0, 1e-6, 1e-3, 1.0],
             biases=[0, 100, 200, 300, 400, 500]),
    30: dict(base_funcs=[bf.elliptic, bf.katsuura, bf.ackley, bf.schwefel, bf.rastrigin, bf.happycat],
             sigmas=[10, 20, 30, 40, 50, 60], lambdas=[1e-6, 1.0, 1.0, 1.0, 1.0, 1.0],
             biases=[0, 100, 200, 300, 400, 500]),
}


def evaluate(fid: int, x: np.ndarray) -> np.ndarray:
    """Evaluate CEC2017 function `fid` (1..30) at points x (..., D)."""
    x = np.asarray(x, dtype=float)
    d = x.shape[-1]
    bias = _BIAS[fid]

    if fid in _UNIMODAL:
        rot = bf.rotation_matrix(d, seed=1022 + fid)
        shift = bf.shift_vector(d, seed=100 + fid)
        z = bf.transform(x, shift, rot)
        return _UNIMODAL[fid](z) + bias

    if fid in _BASIC_MULTIMODAL:
        rot = bf.rotation_matrix(d, seed=1022 + fid)
        shift = bf.shift_vector(d, seed=100 + fid)
        z = bf.transform(x, shift, rot)
        return _BASIC_MULTIMODAL[fid](z) + bias

    if fid in _HYBRID_SPECS:
        funcs, ratios = _HYBRID_SPECS[fid]
        return bf.hybrid(x, 100 + fid, funcs, ratios) + bias

    if fid in _COMPOSITION_SPECS:
        spec = _COMPOSITION_SPECS[fid]
        return bf.composition(x, 100 + fid, **spec) + bias

    raise ValueError(f"Unknown CEC2017 function id: {fid}")


def get_function(fid: int):
    return lambda x: evaluate(fid, x)


def function_name(fid: int) -> str:
    return _NAMES[fid]


def bounds(d: int):
    lb = np.full(d, LOWER_BOUND)
    ub = np.full(d, UPPER_BOUND)
    return lb, ub


ALL_FUNCTION_IDS = list(range(1, 31))
