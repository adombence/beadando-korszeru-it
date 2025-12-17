from __future__ import annotations
from .base import AgentBase

class SARSAAgent(AgentBase):
  """
  On-policy TD control:
  target = r + gamma * Q(s', a')
  """
  def update(self, s: int, a: int, r: float, s_next: int, a_next: int, done: bool) -> float:
    if done:
      target = r
    else:
      target = r + self.cfg.gamma * float(self.Q[s_next, a_next])

    td_error      = target - float(self.Q[s, a])
    self.Q[s, a] += self.cfg.alpha * td_error

    return float(td_error)