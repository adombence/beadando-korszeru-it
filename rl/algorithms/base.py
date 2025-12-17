from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class AgentConfig:
  n_states: int
  n_actions: int
  alpha: float = 0.1
  gamma: float = 0.99
  epsilon: float = 0.1

class AgentBase:
  """
  Egységes agent API Q-learninghez és SARSA-hoz.
  """
  def __init__(self, cfg: AgentConfig):
    self.cfg = cfg
    self.Q = np.zeros((cfg.n_states, cfg.n_actions), dtype=np.float32)

  def act(self, s: int, greedy: bool = False) -> int:
    if greedy or (np.random.rand() > self.cfg.epsilon):
      return int(np.argmax(self.Q[s]))
    return int(np.random.randint(self.cfg.n_actions))
  
  def update(self, s: int, a: int, r: float, s_next: int, a_next: int, done: bool) -> float:
    raise NotImplementedError("Subclasses should implement this method.")