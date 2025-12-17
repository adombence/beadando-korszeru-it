from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rl.training.types import EpisodeLog


@dataclass(frozen=True)
class RunMeta:
    env: str
    agent: str
    seed: int


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def save_episode_logs(logs: list[EpisodeLog], out_root: Path, meta: RunMeta, *, kind: str = "train") -> Path:
    """
    Save per-episode logs for one seed as CSV.

    Columns: ep, return_, length, success, td_error_mean, env, agent, seed
    """
    df = pd.DataFrame(
        {
            "ep": [x.ep for x in logs],
            "return_": [x.return_ for x in logs],
            "length": [x.length for x in logs],
            "success": [int(bool(x.success)) for x in logs],
            "td_error_mean": [
                (np.nan if x.td_error_mean is None else float(x.td_error_mean)) for x in logs
            ],
            "env": meta.env,
            "agent": meta.agent,
            "seed": meta.seed,
        }
    )

    # For backward compatibility, keep training logs as seed_{seed}.csv
    # and write eval logs as eval_seed_{seed}.csv
    if kind == "train":
        filename = f"seed_{meta.seed}.csv"
    else:
        filename = f"{kind}_seed_{meta.seed}.csv"

    out_path = out_root / meta.env / meta.agent / filename
    _ensure_dir(out_path)
    df.to_csv(out_path, index=False)
    return out_path


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr.astype(np.float64)
    s = pd.Series(arr, dtype="float64")
    return s.rolling(window=window, min_periods=1, center=False).mean().to_numpy()


def _logs_to_df(logs: list[EpisodeLog], meta: RunMeta) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ep": [x.ep for x in logs],
            "return_": [x.return_ for x in logs],
            "length": [x.length for x in logs],
            "success": [int(bool(x.success)) for x in logs],
            "td_error_mean": [
                (np.nan if x.td_error_mean is None else float(x.td_error_mean)) for x in logs
            ],
            "env": meta.env,
            "agent": meta.agent,
            "seed": meta.seed,
        }
    )


def _load_seed_files(files: Iterable[Path]) -> list[pd.DataFrame]:
    dfs: list[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            continue
    return dfs


def _aggregate_ma(dfs: list[pd.DataFrame], column: str, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute moving-average per seed, then aggregate across seeds.

    Returns: (x_axis_episodes, mean, std)
    """
    if not dfs:
        return np.array([]), np.array([]), np.array([])

    # Align by episode index (assume equal length across seeds)
    n_ep = int(min(int(df["ep"].max()) + 1 for df in dfs))
    per_seed_ma: list[np.ndarray] = []
    for df in dfs:
        s = df.sort_values("ep").reset_index(drop=True)
        vals = s[column].to_numpy(dtype=float)[:n_ep]
        per_seed_ma.append(moving_average(vals, window))

    mat = np.stack(per_seed_ma, axis=0)  # [n_seed, n_ep]
    mean = mat.mean(axis=0)
    std = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean)
    x = np.arange(n_ep)
    return x, mean, std


def _plot_with_band(x: np.ndarray, mean: np.ndarray, std: np.ndarray, ylabel: str, title: str, out_path: Path) -> None:
    _ensure_dir(out_path)
    plt.figure(figsize=(9, 5))
    plt.plot(x, mean, label="mean", color="#1f77b4")
    if std is not None and std.size:
        plt.fill_between(x, mean - std, mean + std, color="#1f77b4", alpha=0.2, label="±1 std")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_single_run(df: pd.DataFrame, out_dir: Path, env: str, agent: str, window: int = 100) -> list[Path]:
    outputs: list[Path] = []
    if df.empty:
        return outputs

    out_plots = out_dir / "plots"

    if env.startswith("FrozenLake"):
        # Success rate MA
        s = df.sort_values("ep").reset_index(drop=True)
        x = s["ep"].to_numpy(dtype=int)
        succ_ma = moving_average(s["success"].to_numpy(dtype=float), window)
        p1 = out_plots / f"success_rate_ma_w{window}.png"
        _plot_with_band(x, succ_ma, np.array([]), ylabel=f"Success rate (MA w={window})", title=f"{env} | {agent}", out_path=p1)
        outputs.append(p1)

        # Return MA
        ret_ma = moving_average(s["return_"].to_numpy(dtype=float), window)
        p2 = out_plots / f"return_ma_w{window}.png"
        _plot_with_band(x, ret_ma, np.array([]), ylabel=f"Return (MA w={window})", title=f"{env} | {agent}", out_path=p2)
        outputs.append(p2)

    elif env.startswith("CartPole"):
        # Episode length MA
        s = df.sort_values("ep").reset_index(drop=True)
        x = s["ep"].to_numpy(dtype=int)
        len_ma = moving_average(s["length"].to_numpy(dtype=float), window)
        p = out_plots / f"length_ma_w{window}.png"
        _plot_with_band(x, len_ma, np.array([]), ylabel=f"Episode length (MA w={window})", title=f"{env} | {agent}", out_path=p)
        outputs.append(p)

    return outputs


def write_run_outputs(
    logs: list[EpisodeLog],
    out_root: Path,
    meta: RunMeta,
    agent_config: dict,
    train_config: dict,
    eval_summary: dict | None = None,
    window: int = 100,
) -> Path:
    """Save a structured run folder with config, logs, summary and plots.

    Returns the run directory path.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{ts}_{meta.env}_{meta.agent}_seed{meta.seed}"
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Logs
    df = _logs_to_df(logs, meta)
    logs_path = run_dir / "logs.csv"
    df.to_csv(logs_path, index=False)

    # Config
    cfg = {
        "env": meta.env,
        "agent": meta.agent,
        "seed": meta.seed,
        "agent_config": agent_config,
        "train_config": train_config,
        "timestamp": ts,
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Summary (optional, e.g., eval summary)
    if eval_summary is not None:
        with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(eval_summary, f, indent=2)

    # Per-run plots
    _plot_single_run(df, out_dir=run_dir, env=meta.env, agent=meta.agent, window=window)

    return run_dir


def plot_frozenlake(root: Path, agent: str, window: int = 100, env: str = "FrozenLake-v1") -> list[Path]:
    seed_files = sorted((root / env / agent).glob("seed_*.csv"))
    dfs = _load_seed_files(seed_files)

    outputs: list[Path] = []
    if not dfs:
        return outputs

    # Success rate (moving average of success 0/1)
    x, mean, std = _aggregate_ma(dfs, column="success", window=window)
    out1 = root / env / agent / "plots" / f"success_rate_ma_w{window}.png"
    _plot_with_band(x, mean, std, ylabel=f"Success rate (MA w={window})", title=f"{env} | {agent}", out_path=out1)
    outputs.append(out1)

    # Return moving average
    x2, mean2, std2 = _aggregate_ma(dfs, column="return_", window=window)
    out2 = root / env / agent / "plots" / f"return_ma_w{window}.png"
    _plot_with_band(x2, mean2, std2, ylabel=f"Return (MA w={window})", title=f"{env} | {agent}", out_path=out2)
    outputs.append(out2)

    return outputs


def plot_cartpole(root: Path, agent: str, window: int = 100) -> list[Path]:
    env = "CartPole-v1"
    seed_files = sorted((root / env / agent).glob("seed_*.csv"))
    dfs = _load_seed_files(seed_files)

    outputs: list[Path] = []
    if not dfs:
        return outputs

    # Episode length moving average (classic CartPole metric)
    x, mean, std = _aggregate_ma(dfs, column="length", window=window)
    out = root / env / agent / "plots" / f"length_ma_w{window}.png"
    _plot_with_band(x, mean, std, ylabel=f"Episode length (MA w={window})", title=f"{env} | {agent}", out_path=out)
    outputs.append(out)

    return outputs
