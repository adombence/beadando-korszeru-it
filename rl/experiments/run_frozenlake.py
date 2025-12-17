from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

# Ensure project root on sys.path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.adapter.frozenlake import FrozenLakeAdapter
from rl.algorithms.base import AgentConfig
from rl.algorithms.q_learning import QLearningAgent
from rl.algorithms.sarsa import SARSAAgent
from rl.experiments.common import evaluate, seed_everything, summarize, train


def run(agent_cls, name: str):
  base_env = gym.make("FrozenLake-v1", is_slippery=False)
  env = FrozenLakeAdapter(base_env)

  cfg = AgentConfig(
    n_states=env.n_states,
    n_actions=env.n_actions,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
  )

  seeds = [0, 1, 2, 3, 4]
  train_episodes = 3000
  eval_episodes  = 300
  max_steps      = 200

  per_seed = []
  for seed in seeds:
    seed_everything(seed)
    base_env.reset(seed=seed)

    agent = agent_cls(cfg)
    _ = train(env, agent, episodes=train_episodes, max_steps=max_steps)
    eval_logs = evaluate(env, agent, episodes=eval_episodes, max_steps=max_steps)
    per_seed.append(summarize(eval_logs))

  sr = np.array([x["success_rate"] for x in per_seed], dtype=np.float32)
  rm = np.array([x["return_mean"]  for x in per_seed], dtype=np.float32)
  lm = np.array([x["length_mean"]  for x in per_seed], dtype=np.float32)

  print(f"\n=== {name} | FrozenLake-v1 ===")
  print(f"Success rate: {sr.mean():.3f} ± {sr.std(ddof=1):.3f}")
  print(f"Return mean : {rm.mean():.3f} ± {rm.std(ddof=1):.3f}")
  print(f"Length mean : {lm.mean():.3f} ± {lm.std(ddof=1):.3f}")

def main():
  run(QLearningAgent, "Q-learning")
  run(SARSAAgent, "SARSA")

if __name__ == "__main__":
  main()
