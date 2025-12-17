from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeLog:
    ep: int
    return_: float
    length: int
    success: bool
    td_error_mean: float | None = None
