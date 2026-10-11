"""
Generic Q-learning based strategy-selection agent used by RELEASE.

This module is shared by all three RELEASE variants (GA, SA, QiGA). It is
deliberately algorithm-agnostic: the host optimizer

  1. computes a *state* (a discrete index 0..N_STATES-1) describing the
     current search behaviour (improvement trend + diversity),
  2. asks the agent to pick an *action* (a discrete index 0..N_ACTIONS-1)
     describing a point on the "intensification <-> diversification"
     strategy spectrum (A1..A9 in the paper's figure),
  3. maps that action to concrete operator hyper-parameters (this mapping
     is algorithm-specific and lives in each algorithm's module),
  4. runs one iteration / generation,
  5. computes a scalar reward from the resulting fitness improvement,
  6. calls agent.update(state, action, reward, next_state).

The agent uses tabular Q-learning with an epsilon-greedy / softmax policy,
matching the "Softmax and epsilon-greedy" block in the RELEASE flowchart.
"""

from __future__ import annotations

import numpy as np

N_STATES = 27    # 3 (improvement trend) x 3 (diversity) x 3 (search phase)
N_ACTIONS = 9    # A1 (intensification) ... A9 (diversification)


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    x = x / max(temperature, 1e-8)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy + softmax action policy."""

    def __init__(
        self,
        n_states: int = N_STATES,
        n_actions: int = N_ACTIONS,
        alpha: float = 0.15,
        gamma: float = 0.90,
        epsilon_start: float = 0.30,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        temperature: float = 0.5,
        rng: np.random.Generator | None = None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.temperature = temperature
        self.Q = np.zeros((n_states, n_actions))
        self.rng = rng if rng is not None else np.random.default_rng()

    def select_action(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        probs = _softmax(self.Q[state], self.temperature)
        return int(self.rng.choice(self.n_actions, p=probs))

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        best_next = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class StateEncoder:
    """
    Encodes the search state as one of N_STATES = 3 x 3 x 3 discrete bins:

      - improvement trend (relative improvement of the best fitness over a
        sliding window): stagnant / moderate / fast  -> bins 0,1,2
      - diversity (normalized spread of the population / recent moves):
        low / medium / high -> bins 0,1,2
      - search phase (fraction of the evaluation budget used so far):
        early / mid / late -> bins 0,1,2

    state index = phase_bin * 9 + trend_bin * 3 + diversity_bin

    The phase feature lets the agent learn phase-dependent policies (e.g.
    favour diversification early and intensification late) directly,
    independent of / on top of the epoch-length growth schedule.
    """

    def __init__(self, window: int = 5,
                 trend_thresholds=(1e-6, 1e-3),
                 diversity_thresholds=(0.05, 0.20)):
        self.window = window
        self.trend_thresholds = trend_thresholds
        self.diversity_thresholds = diversity_thresholds
        self.history: list[float] = []

    def reset(self):
        self.history = []

    def trend_bin(self, current_best: float) -> int:
        self.history.append(current_best)
        if len(self.history) < 2:
            return 0
        window = self.history[-self.window - 1:]
        prev = window[0]
        cur = window[-1]
        denom = abs(prev) + 1e-12
        rel_improve = (prev - cur) / denom  # positive = improvement (minimization)
        rel_improve = max(rel_improve, 0.0)
        lo, hi = self.trend_thresholds
        if rel_improve < lo:
            return 0
        elif rel_improve < hi:
            return 1
        else:
            return 2

    @staticmethod
    def diversity_bin(diversity: float, thresholds=(0.05, 0.20)) -> int:
        lo, hi = thresholds
        if diversity < lo:
            return 0
        elif diversity < hi:
            return 1
        else:
            return 2

    @staticmethod
    def phase_bin(progress: float, thresholds=(1.0 / 3.0, 2.0 / 3.0)) -> int:
        lo, hi = thresholds
        if progress < lo:
            return 0
        elif progress < hi:
            return 1
        else:
            return 2

    def encode(self, current_best: float, diversity: float, progress: float = 0.0) -> int:
        t = self.trend_bin(current_best)
        d = self.diversity_bin(diversity, self.diversity_thresholds)
        p = self.phase_bin(progress)
        return p * 9 + t * 3 + d


def relative_improvement_reward(prev_best: float, new_best: float,
                                 stagnation_penalty: float = -0.001) -> float:
    """Reward = relative improvement of best fitness (minimization).

    Positive when the new best is better, a small fixed penalty when there
    is no improvement (encourages the agent to move away from
    non-productive strategies / states).
    """
    denom = abs(prev_best) + 1e-12
    improvement = (prev_best - new_best) / denom
    if improvement <= 0:
        return stagnation_penalty
    return float(improvement)


# --------------------------------------------------------------------------
# Shared 9-point "intensification <-> diversification" strategy spectrum
# --------------------------------------------------------------------------

def spectrum(low: float, high: float, n: int = N_ACTIONS) -> np.ndarray:
    """Return n values evenly spaced from `low` (A1) to `high` (A9)."""
    return np.linspace(low, high, n)
