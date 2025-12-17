from __future__ import annotations

from .base import AgentBase


class QLearningAgent(AgentBase):
    """
    Off-policy TD Control:
    target = r + gamma * max_a' Q(s', a')
    """

    def update(self, s: int, a: int, r: float, s_next: int, a_next: int, done: bool) -> float:
        target = r if done else r + self.cfg.gamma * float(self.Q[s_next].max())
        td_error = target - float(self.Q[s, a])
        self.Q[s, a] += self.cfg.alpha * td_error

        return float(td_error)
