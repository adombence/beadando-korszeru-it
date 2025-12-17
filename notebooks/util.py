from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path("../results")

def load_logs(env_name: str, agent_name: str) -> pd.DataFrame:
    """
    Betölti az összes seed logját egy adott environment + agent párosra.
    """
    dfs = []
    pattern = f"*_{env_name}_{agent_name}_seed*/logs.csv"
    for path in RESULTS_DIR.glob(pattern):
        df = pd.read_csv(path)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No results found for {env_name} + {agent_name}")
    return pd.concat(dfs, ignore_index=True)

def moving_average(x, window=100):
    return np.convolve(x, np.ones(window) / window, mode="valid")

def aggregate_by_episode(df: pd.DataFrame, column: str):
    grouped = df.groupby("ep")[column]
    return grouped.mean(), grouped.std()