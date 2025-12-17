from typing import Any

import gymnasium as gym
from rl.adapter.cartpole import CartPoleAdapter
from rl.adapter.frozenlake import FrozenLakeAdapter


def _check_types(step_out: tuple[int, float, bool, dict[str, Any]]):
    s_next, r, done, info = step_out
    assert isinstance(s_next, int)
    assert isinstance(r, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_frozenlake_adapter_reset_and_step_types():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    adapter = FrozenLakeAdapter(env)

    s0 = adapter.reset(seed=0)
    assert isinstance(s0, int)
    assert 0 <= s0 < adapter.n_states

    out = adapter.step(0)
    _check_types(out)

    # success flag must be present in info
    _, _, _, info = out
    assert "success" in info and isinstance(info["success"], bool)


def test_cartpole_adapter_reset_and_step_types():
    env = gym.make("CartPole-v1")
    adapter = CartPoleAdapter(env)

    s0 = adapter.reset(seed=42)
    assert isinstance(s0, int)
    assert 0 <= s0 < adapter.n_states

    out = adapter.step(0)
    _check_types(out)
