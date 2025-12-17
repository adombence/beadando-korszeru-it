from __future__ import annotations

from typing import Any

import gymnasium as gym


class GymDiscreteAdapter:
    def __init__(self, env: gym.Env):
        self.env = env
        if not hasattr(env.observation_space, "n"):
            raise ValueError("GymDiscreteAdapter expects a Discrete observation space.")
        if not hasattr(env.action_space, "n"):
            raise ValueError("GymDiscreteAdapter expects a Discrete action space.")
        self.n_states = int(env.observation_space.n)
        self.n_actions = int(env.action_space.n)

    def reset(self, seed: int | None = None) -> int:
        obs, _info = self.env.reset(seed=seed)
        return int(obs)

    def step(self, action: int) -> tuple[int, float, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        done = bool(terminated or truncated)
        info = dict(info) if info is not None else {}
        return int(obs), float(reward), done, info
