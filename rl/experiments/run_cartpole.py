from __future__ import annotations
import gymnasium as gym
import numpy as np

from algorithms.base import AgentConfig
from algorithms.q_learning import QLearningAgent
from algorithms.sarsa import SARSAAgent

from adapter.cartpole import CartPoleAdapter, CartPoleConfig
from experiments.common import train, evaluate, summarize, seed_everything

def run(agent_cls, name: str):
    base_env = gym.make("CartPole-v1")
    env = CartPoleAdapter(base_env, cfg=CartPoleConfig())

    cfg = AgentConfig(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1,
    )

    seeds = [0, 1, 2, 3, 4]
    train_episodes = 2000
    eval_episodes = 50
    max_steps = 500  # CartPole standard episode cap

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

    print(f"\n=== {name} | CartPole-v1 (discretized) ===")
    print(f"Success rate: {sr.mean():.3f} ± {sr.std(ddof=1):.3f}   (success ~ reached max_steps)")
    print(f"Return mean : {rm.mean():.1f} ± {rm.std(ddof=1):.1f}")
    print(f"Length mean : {lm.mean():.1f} ± {lm.std(ddof=1):.1f}")

def main():
    run(QLearningAgent, "Q-learning")
    run(SARSAAgent, "SARSA")

if __name__ == "__main__":
    main()
