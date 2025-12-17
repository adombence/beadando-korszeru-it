# Classical RL Experiments (Q-learning, SARSA)

Small, self-contained experiments for value-based RL on Gymnasium environments with discrete adapters, per-episode logging, learning curves, and multi-seed aggregation.

## Install

Tested on macOS with Python 3.9+.

```bash
# From project root
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Experiments

Two ready-made scripts train and evaluate Q-learning and SARSA with multiple seeds, write structured results, and generate learning curves.

```bash
# From project root
python -u ./rl/experiments/run_frozenlake.py
python -u ./rl/experiments/run_cartpole.py
```

- Environments
  - FrozenLake-v1 (deterministic, is_slippery=False)
  - CartPole-v1 (discretized observations via adapter)
- Algorithms: Q-learning, SARSA

Edit the episode counts, seeds, and moving-average window directly in:
- FrozenLake: [rl/experiments/run_frozenlake.py](rl/experiments/run_frozenlake.py)
- CartPole: [rl/experiments/run_cartpole.py](rl/experiments/run_cartpole.py)

## Results Layout

All outputs go under `results/` in two forms:

1) Per-seed structured “run package” (timestamped)
- results/YYYYMMDD_HHMMSS_<env>_<agent>_seed<seed>/
  - config.json: env, agent, seed, agent_config, train_config, timestamp
  - logs.csv: per-episode logs for that seed
  - summary.json: evaluation summary for that seed
  - plots/: single-run learning curves (moving averages)

2) Aggregate per-env/per-agent folder (across seeds)
- results/<env>/<agent>/seed_0.csv, seed_1.csv, ...
- results/<env>/<agent>/plots/
  - FrozenLake: success_rate_ma_w<window>.png, return_ma_w<window>.png
  - CartPole: length_ma_w<window>.png

## Metrics

Per-episode (saved in logs.csv):
- ep: episode index (0-based)
- return_: cumulative reward over the episode
- length: steps taken in the episode
- success: task success flag (FrozenLake: goal reached; CartPole: reaching max steps is treated as success)
- td_error_mean: mean TD error during the episode

Aggregates (per-seed evaluation summary and multi-seed plots):
- return_mean, return_std
- length_mean, length_std
- success_rate: fraction of successful episodes
- Learning curves: moving averages per episode with window (default 100); multi-seed plots include mean ± std band across seeds.

## Reproducibility & Policies

- Seeding
  - Each seed run calls a global seed helper and resets the base Gym env with `seed`.
  - Agents use their own RNG: `np.random.default_rng(seed)` for deterministic behavior.
  - Optional per-episode seeding exists in training (disabled by default), which would reset with `seed + episode`.
- Action selection
  - Training uses explicit `policy="epsilon_greedy"`.
  - Evaluation uses `policy="greedy"`.
  - Epsilon can be overridden per call; default comes from `AgentConfig`.

## Where Things Live

- Adapters and Protocols
  - Discrete adapters: [rl/adapter/](rl/adapter/)
  - Adapter protocol: [rl/adapter/protocols.py](rl/adapter/protocols.py)
- Algorithms
  - Base + agents: [rl/algorithms/](rl/algorithms/)
- Training & Evaluation utils
  - Common loop + summaries: [rl/experiments/common.py](rl/experiments/common.py)
  - Standalone trainer (optional): [rl/training/train.py](rl/training/train.py)
- Plotting & saving
  - Utilities for logging and figures: [rl/experiments/plotting.py](rl/experiments/plotting.py)

## Tips

- If you want faster quick-checks, reduce `train_episodes`/`eval_episodes` and the `seeds` list in the run scripts.
- If you want a different moving-average window for plots, change `window` in the run scripts and in `write_run_outputs()` calls.
- To try different exploration, adjust `AgentConfig.epsilon` or pass `epsilon=` to `agent.act(..., policy="epsilon_greedy", epsilon=...)`.
