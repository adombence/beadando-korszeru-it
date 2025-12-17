from __future__ import annotations

import sys
from dataclasses import asdict
import argparse
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
from rl.experiments.plotting import RunMeta, plot_frozenlake, save_episode_logs, write_run_outputs


def run(agent_cls, name: str, *, slippery: bool, seeds: list[int], train_episodes: int, eval_episodes: int, max_steps: int, window: int):
    base_env = gym.make("FrozenLake-v1", is_slippery=slippery)
    env = FrozenLakeAdapter(base_env)

    env_name = "FrozenLake-v1-slippery" if slippery else "FrozenLake-v1"
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
            epsilon=1.0,
            seed=seed,
        )
        agent = agent_cls(cfg)
        train_logs = train(env, agent, episodes=train_episodes, max_steps=max_steps)
        # Save per-episode training logs for learning curves
        save_episode_logs(
            train_logs,
            out_root=out_root,
            meta=RunMeta(env=env_name, agent=name, seed=seed),
            kind="train",
        )
        eval_logs = evaluate(env, agent, episodes=eval_episodes, max_steps=max_steps)
        summary = summarize(eval_logs)
        # Save eval per-episode logs as well
        save_episode_logs(
            eval_logs,
            out_root=out_root,
            meta=RunMeta(env=env_name, agent=name, seed=seed),
            kind="eval",
        )
        per_seed.append(summary)

        # Structured run outputs (config, logs, summary, per-run plots)
        _ = write_run_outputs(
            logs=train_logs,
            out_root=out_root,
            meta=RunMeta(env=env_name, agent=name, seed=seed),
            agent_config=asdict(cfg),
            train_config={
                "train_episodes": train_episodes,
                "eval_episodes": eval_episodes,
                "max_steps": max_steps,
                "policy_train": "epsilon_greedy",
                "policy_eval": "greedy",
                "ma_window": window,
            },
            eval_summary=summary,
            window=window,
        )

    sr = np.array([x["success_rate"] for x in per_seed], dtype=np.float32)
    rm = np.array([x["return_mean"] for x in per_seed], dtype=np.float32)
    lm = np.array([x["length_mean"] for x in per_seed], dtype=np.float32)

    print(f"\n=== {name} | {env_name} ===")
    print(f"Success rate: {sr.mean():.3f} ± {sr.std(ddof=1):.3f}")
    print(f"Return mean : {rm.mean():.3f} ± {rm.std(ddof=1):.3f}")
    print(f"Length mean : {lm.mean():.3f} ± {lm.std(ddof=1):.3f}")

    # Create learning curve plots: success rate + return moving averages
    plots = plot_frozenlake(out_root, agent=name, window=window, env=env_name)
    for p in plots:
        print(f"Saved plot: {p}")


def main():
    parser = argparse.ArgumentParser(description="Run FrozenLake experiments with logging and plots")
    parser.add_argument("--slippery", action="store_true", help="Use slippery FrozenLake variant")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds (0..N-1)")
    parser.add_argument("--train-episodes", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--window", type=int, default=100, help="Moving average window")
    args = parser.parse_args()

    seeds = list(range(int(args.seeds)))
    run(QLearningAgent, "Q-learning", slippery=bool(args.slippery), seeds=seeds, train_episodes=int(args.train_episodes), eval_episodes=int(args.eval_episodes), max_steps=int(args.max_steps), window=int(args.window))
    run(SARSAAgent, "SARSA", slippery=bool(args.slippery), seeds=seeds, train_episodes=int(args.train_episodes), eval_episodes=int(args.eval_episodes), max_steps=int(args.max_steps), window=int(args.window))


if __name__ == "__main__":
    main()
