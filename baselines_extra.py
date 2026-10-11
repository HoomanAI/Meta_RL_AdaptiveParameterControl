"""Additional adaptive baselines used as competitors:
SHADE (Tanabe & Fukunaga, 2013) and LDIW-PSO (Shi & Eberhart, 1998/1999)."""
import numpy as np

from algorithms.common import clip_to_bounds, init_population, pso_step


def run_shade(func, d, lb, ub, pop_size=30, max_evals=10000, seed=None, H=None, trace=False):
    rng = np.random.default_rng(seed)
    H = H or pop_size
    pop = init_population(pop_size, d, lb, ub, rng)
    fit = func(pop)
    evals = pop_size
    MF, MCR = np.full(H, 0.5), np.full(H, 0.5)
    k = 0
    archive = np.empty((0, d))
    hist = [float(fit.min())]
    while evals < max_evals:
        r = rng.integers(0, H, pop_size)
        CR = np.clip(rng.normal(MCR[r], 0.1), 0, 1)
        F = np.empty(pop_size)
        for i in range(pop_size):
            f = -1.0
            while f <= 0:
                f = MF[r[i]] + 0.1 * np.tan(np.pi * (rng.random() - 0.5))
            F[i] = min(f, 1.0)
        p = rng.uniform(2.0 / pop_size, 0.2, pop_size)
        order = np.argsort(fit)
        union = np.vstack([pop, archive]) if len(archive) else pop
        trial = np.empty_like(pop)
        for i in range(pop_size):
            top = order[:max(2, int(round(p[i] * pop_size)))]
            pb = pop[rng.choice(top)]
            r1 = rng.choice([j for j in range(pop_size) if j != i])
            r2 = rng.integers(0, len(union))
            while r2 == i or r2 == r1:
                r2 = rng.integers(0, len(union))
            v = pop[i] + F[i] * (pb - pop[i]) + F[i] * (pop[r1] - union[r2])
            v = np.where(v < lb, (lb + pop[i]) / 2, np.where(v > ub, (ub + pop[i]) / 2, v))
            mask = rng.random(d) < CR[i]
            mask[rng.integers(d)] = True
            trial[i] = np.where(mask, v, pop[i])
        n = min(pop_size, max_evals - evals)
        tf = func(trial[:n])
        evals += n
        better = tf < fit[:n]
        if np.any(better):
            idx = np.flatnonzero(better)
            w = np.abs(fit[idx] - tf[better])
            w = w / w.sum()
            archive = np.vstack([archive, pop[idx]])
            if len(archive) > pop_size:
                archive = archive[rng.choice(len(archive), pop_size, replace=False)]
            sF, sCR = F[idx], CR[idx]
            MF[k] = np.sum(w * sF ** 2) / max(np.sum(w * sF), 1e-12)
            MCR[k] = np.sum(w * sCR)
            k = (k + 1) % H
            pop[idx] = trial[idx]
            fit[idx] = tf[better]
        hist.append(float(fit.min()))
    return (float(fit.min()), hist) if trace else float(fit.min())


def run_ldiw_pso(func, d, lb, ub, pop_size=30, max_evals=10000, seed=None, trace=False):
    rng = np.random.default_rng(seed)
    span = ub - lb
    v_max = 0.2 * span
    pos = init_population(pop_size, d, lb, ub, rng)
    vel = rng.uniform(-1.0, 1.0, size=(pop_size, d)) * 0.1 * span
    fit = func(pos)
    evals = pop_size
    pbp, pbf = pos.copy(), fit.copy()
    i = int(np.argmin(fit))
    gbp, gbf = pos[i].copy(), float(fit[i])
    hist = [gbf]
    while evals < max_evals:
        w = 0.9 - 0.5 * evals / max_evals
        pos, vel, pbp, pbf, gbp, gbf, n = pso_step(func, pos, vel, pbp, pbf, gbp, gbf, lb, ub,
                                                    w, 2.0, 2.0, v_max, max_evals - evals, rng)
        evals += n
        hist.append(float(gbf))
    return (float(gbf), hist) if trace else float(gbf)
