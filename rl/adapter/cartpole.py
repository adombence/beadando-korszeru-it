from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class CartPoleConfig:
    # bins per dimension (x, x_dot, theta, theta_dot)
    bins: tuple[int, int, int, int] = (6, 6, 12, 12)

    # clipping ranges (avoid huge values)
    x_range: tuple[float, float] = (-2.4, 2.4)
    x_dot_range: tuple[float, float] = (-3.0, 3.0)
    theta_range: tuple[float, float] = (-0.418, 0.418)  # ~24 degrees (rad)
    theta_dot_range: tuple[float, float] = (-3.5, 3.5)


class CartPoleAdapter:
    """
    Discretizes CartPole-v1 continuous obs into a single discrete state id.
    Compatible with: reset() -> int, step(a) -> (int, float, bool, info)
    """

    def __init__(self, env: gym.Env, cfg: CartPoleConfig = CartPoleConfig()):  # noqa: B008
        self.env = env
        self.cfg = cfg

        self.n_actions = int(env.action_space.n)
        self._bins = np.array(cfg.bins, dtype=np.int32)

        # edges per dimension for digitize (bins-1 internal edges)
        ranges = [cfg.x_range, cfg.x_dot_range, cfg.theta_range, cfg.theta_dot_range]
        self._edges = [
            np.linspace(lo, hi, num=b + 1, endpoint=True)[1:-1]  # internal cut points
            for (lo, hi), b in zip(ranges, self._bins)
        ]

        # base multipliers to map 4D index -> scalar id
        # idx = [i0,i1,i2,i3], id = i0*B1*B2*B3 + i1*B2*B3 + i2*B3 + i3
        b0, b1, b2, b3 = [int(x) for x in self._bins]
        self._bases = np.array([b1 * b2 * b3, b2 * b3, b3, 1], dtype=np.int32)

        self.n_states = int(b0 * b1 * b2 * b3)

    def _clip_obs(self, obs: np.ndarray) -> np.ndarray:
        x, x_dot, theta, theta_dot = obs
        x = float(np.clip(x, *self.cfg.x_range))
        x_dot = float(np.clip(x_dot, *self.cfg.x_dot_range))
        theta = float(np.clip(theta, *self.cfg.theta_range))
        theta_dot = float(np.clip(theta_dot, *self.cfg.theta_dot_range))
        return np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

    def _discretize(self, obs: np.ndarray) -> int:
        obs = self._clip_obs(obs)
        idxs = [int(np.digitize(v, e)) for v, e in zip(obs, self._edges)]  # 0..bins-1
        return int(np.dot(np.array(idxs, dtype=np.int32), self._bases))

    def reset(self, seed: int | None = None) -> int:
        obs, _info = self.env.reset(seed=seed)
        return self._discretize(np.asarray(obs, dtype=np.float32))

    def step(self, action: int) -> tuple[int, float, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        done = bool(terminated or truncated)
        s_next = self._discretize(np.asarray(obs, dtype=np.float32))
        info = dict(info) if info is not None else {}

        return s_next, float(reward), done, info
