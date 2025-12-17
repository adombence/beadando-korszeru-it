from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np

# Ensure project root on sys.path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.adapter.cartpole import CartPoleAdapter, CartPoleConfig
from rl.algorithms.base import AgentConfig
from rl.algorithms.q_learning import QLearningAgent
from rl.algorithms.sarsa import SARSAAgent
from rl.experiments.common import evaluate, seed_everything, summarize, train
from rl.experiments.plotting import RunMeta, plot_cartpole, save_episode_logs, write_run_outputs


def run(agent_cls, name: str):
    base_env = gym.make("CartPole-v1")
    env = CartPoleAdapter(base_env, cfg=CartPoleConfig())

    seeds = [0, 1, 2, 3, 4]
    train_episodes = 2000
    eval_episodes = 50
    max_steps = 500  # CartPole standard episode cap
    out_root = Path("results")

    per_seed = []
    for seed in seeds:
        seed_everything(seed)
        base_env.reset(seed=seed)

        cfg = AgentConfig(
            n_states=env.n_states,
            n_actions=env.n_actions,
            alpha=0.1,
            gamma=0.99,
            epsilon=0.1,
            seed=seed,
        )
        agent = agent_cls(cfg)
        train_logs = train(env, agent, episodes=train_episodes, max_steps=max_steps)
        # Save per-episode training logs for learning curves
        save_episode_logs(
            train_logs,
            out_root=out_root,
            meta=RunMeta(env="CartPole-v1", agent=name, seed=seed),
        )
        eval_logs = evaluate(env, agent, episodes=eval_episodes, max_steps=max_steps)
        summary = summarize(eval_logs)
        per_seed.append(summary)

        # Structured run outputs (config, logs, summary, per-run plots)
        _ = write_run_outputs(
            logs=train_logs,
            out_root=out_root,
            meta=RunMeta(env="CartPole-v1", agent=name, seed=seed),
            agent_config=asdict(cfg),
            train_config={
                "train_episodes": train_episodes,
                "eval_episodes": eval_episodes,
                "max_steps": max_steps,
                "policy_train": "epsilon_greedy",
                "policy_eval": "greedy",
                "ma_window": 100,
            },
            eval_summary=summary,
            window=100,
        )

    sr = np.array([x["success_rate"] for x in per_seed], dtype=np.float32)
    rm = np.array([x["return_mean"] for x in per_seed], dtype=np.float32)
    lm = np.array([x["length_mean"] for x in per_seed], dtype=np.float32)

    print(f"\n=== {name} | CartPole-v1 (discretized) ===")
    print(f"Success rate: {sr.mean():.3f} ± {sr.std(ddof=1):.3f}   (success ~ reached max_steps)")
    print(f"Return mean : {rm.mean():.1f} ± {rm.std(ddof=1):.1f}")
    print(f"Length mean : {lm.mean():.1f} ± {lm.std(ddof=1):.1f}")

    # Create learning curve plots (episode length moving average)
    plots = plot_cartpole(out_root, agent=name, window=100)
    for p in plots:
        print(f"Saved plot: {p}")


def main():
    run(QLearningAgent, "Q-learning")
    run(SARSAAgent, "SARSA")


if __name__ == "__main__":
    main()
