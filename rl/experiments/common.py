from __future__ import annotations

import numpy as np
from rl.adapter import EnvAdapter
from rl.training.types import EpisodeLog


def _infer_success(done: bool, reward: float, t: int, max_steps: int, info: dict) -> bool:
    if isinstance(info, dict) and "success" in info:
        return bool(info["success"])
    if done and reward > 0.0:
        return True
    return t + 1 >= max_steps


def train(env: EnvAdapter, agent, episodes: int, max_steps: int, episode_seed_base: int | None = None) -> list[EpisodeLog]:
    logs: list[EpisodeLog] = []

    for ep in range(episodes):
        # Optional per-episode deterministic seeding
        s = env.reset(seed=(episode_seed_base + ep) if episode_seed_base is not None else None)
        a = agent.act(s, policy="epsilon_greedy")

        ep_return = 0.0
        td_errors: list[float] = []
        success = False

        for t in range(max_steps):
            s_next, r, done, info = env.step(a)
            ep_return += float(r)

            a_next = agent.act(s_next, policy="epsilon_greedy") if not done else 0
            td = agent.update(s, a, float(r), s_next, a_next, bool(done))
            td_errors.append(float(td))

            success = _infer_success(bool(done), float(r), t, max_steps, info)

            s, a = s_next, a_next
            if done:
                break

        logs.append(
            EpisodeLog(
                ep=ep,
                return_=ep_return,
                length=t + 1,
                success=success,
                td_error_mean=float(np.mean(td_errors)) if td_errors else 0.0,
            )
        )

    return logs


def evaluate(env: EnvAdapter, agent, episodes: int, max_steps: int) -> list[EpisodeLog]:
    logs: list[EpisodeLog] = []

    for ep in range(episodes):
        s = env.reset()
        ep_return = 0.0
        success = False

        for t in range(max_steps):
            a = agent.act(s, policy="greedy")
            s, r, done, info = env.step(a)
            ep_return += float(r)

            success = _infer_success(bool(done), float(r), t, max_steps, info)

            if done:
                break

        logs.append(EpisodeLog(ep=ep, return_=ep_return, length=t + 1, success=success))

    return logs


def summarize(logs: list[EpisodeLog]) -> dict:
    rets = np.array([x.return_ for x in logs], dtype=np.float32)
    lens = np.array([x.length for x in logs], dtype=np.float32)
    succ = np.array([x.success for x in logs], dtype=np.float32)
    return {
        "return_mean": float(rets.mean()),
        "return_std": float(rets.std(ddof=1)) if len(rets) > 1 else 0.0,
        "length_mean": float(lens.mean()),
        "length_std": float(lens.std(ddof=1)) if len(lens) > 1 else 0.0,
        "success_rate": float(succ.mean()),
        "n": int(len(logs)),
    }


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
