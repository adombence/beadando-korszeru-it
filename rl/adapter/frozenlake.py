from typing import Any

import gymnasium as gym


class FrozenLakeAdapter:
    def __init__(self, env: gym.Env):
        self.env = env
        try:
            self.n_states = int(env.observation_space.n)
        except Exception as e:
            raise ValueError("FrozenLake adapter expects a Discrete observation space.") from e
        self.n_actions = int(env.action_space.n)

    def reset(self) -> int:
        obs, _info = self.env.reset()
        return int(obs)

    def step(self, action: int) -> tuple[int, float, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        done = bool(terminated or truncated)
        info = dict(info) if info is not None else {}
        info["success"] = bool(done and float(reward) > 0.0)
        return int(obs), float(reward), done, info
