from __future__ import annotations

from .types import EpisodeLog


def evaluate(env, agent, episodes: int, max_steps: int) -> list[EpisodeLog]:
    logs: list[EpisodeLog] = []

    for ep in range(episodes):
        s = env.reset()
        ep_return = 0.0
        success = False

        for _t in range(max_steps):
            a = agent.act(s, greedy=True)
            s, r, done, info = env.step(a)
            ep_return += float(r)

            if "success" in info:
                success = bool(info["success"])

            if done:
                break

        if not success and (_t + 1) >= max_steps:
            success = True

        logs.append(EpisodeLog(ep=ep, return_=ep_return, length=_t + 1, success=success))

    return logs
