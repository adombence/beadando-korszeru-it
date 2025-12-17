from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpisodeLog:
    ep: int
    return_: float
    length: int
    success: bool
    td_error_mean: float | None = None


def train(env, agent, episodes: int, max_steps: int) -> list[EpisodeLog]:
    logs: list[EpisodeLog] = []

    for ep in range(episodes):
        s = env.reset()
        a = agent.act(s, greedy=False)

        ep_return = 0.0
        td_errors: list[float] = []
        success = False

        for _t in range(max_steps):
            s_next, r, done, info = env.step(a)
            ep_return += float(r)

            a_next = agent.act(s_next, greedy=False) if not done else 0

            td = agent.update(s, a, float(r), s_next, a_next, done)
            td_errors.append(float(td))

            if "success" in info:
                success = bool(info["success"])

            s, a = s_next, a_next
            if done:
                break

        if not success and (_t + 1) >= max_steps:
            success = True

        logs.append(
            EpisodeLog(
                ep=ep,
                return_=ep_return,
                length=_t + 1,
                success=success,
                td_error_mean=float(np.mean(td_errors)) if td_errors else 0.0,
            )
        )

    return logs
