from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class AgentConfig:
    n_states: int
    n_actions: int
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 0.1
    seed: int | None = None


class AgentBase:
    """
    Egységes agent API Q-learninghez és SARSA-hoz.
    """

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.Q = np.zeros((cfg.n_states, cfg.n_actions), dtype=np.float32)
        self.rng = np.random.default_rng(cfg.seed)


    def _act_greedy(self, state: int) -> int:
        return int(np.argmax(self.Q[state]))

    def _act_epsilon_greedy(self, state: int, epsilon: float | None = None) -> int:
        eps = self.cfg.epsilon if epsilon is None else float(epsilon)
        if float(self.rng.random()) < eps:
            return int(self.rng.integers(self.cfg.n_actions))
        return int(np.argmax(self.Q[state]))

    def act(
        self,
        state: int,
        *,
        policy: Literal["greedy", "epsilon_greedy"] | None = None,
        epsilon: float | None = None,
        greedy: bool | None = None,
    ) -> int:
        """
        Select an action using the specified policy.

        - policy: "greedy" or "epsilon_greedy" (preferred explicit control)
        - epsilon: optional override for epsilon when using epsilon_greedy
        - greedy: legacy flag; if provided and policy is None, it maps to
                  policy="greedy" when True else "epsilon_greedy".
        """
        pol = policy
        if pol is None:
            pol = "greedy" if bool(greedy) else "epsilon_greedy"

        if pol == "greedy":
            return self._act_greedy(state)
        if pol == "epsilon_greedy":
            return self._act_epsilon_greedy(state, epsilon=epsilon)
        # Fallback to greedy if unknown policy
        return self._act_greedy(state)

    def update(self, s: int, a: int, r: float, s_next: int, a_next: int, done: bool) -> float:
        raise NotImplementedError("Subclasses should implement this method.")
