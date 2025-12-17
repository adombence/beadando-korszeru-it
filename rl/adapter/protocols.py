from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EnvAdapter(Protocol):
    """Minimal adapter interface used by training/evaluation.

    Implementations should expose discrete state/action sizes and
    discrete transitions with (int state, float reward, bool done, info dict).
    """

    # Discrete space sizes
    n_states: int
    n_actions: int

    def reset(self, seed: int | None = None) -> int:
        """Reset environment and return initial discrete state id."""
        ...

    def step(self, action: int) -> tuple[int, float, bool, dict[str, Any]]:
        """Advance one step and return (next_state, reward, done, info)."""
        ...
