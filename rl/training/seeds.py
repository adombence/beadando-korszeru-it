from __future__ import annotations
import random
import numpy as np

def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def reset_env_seed(env, seed: int) -> None:
    # Gymnasium: env.reset(seed=...)
    env.reset(seed=seed)
