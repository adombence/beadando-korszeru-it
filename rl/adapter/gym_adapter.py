import gymnasium as gym

class GymDiscreteAdapter:
  def __init__(self, env: gym.Env):
    self.env        = env
    if not hasattr(env.observation_space, "n"):
      raise ValueError("GymDiscreteAdapter expects a Discrete observation space.")
    if not hasattr(env.action_space, "n"):
      raise ValueError("GymDiscreteAdapter expects a Discrete action space.")
    self.n_states   = int(env.observation_space.n)
    self.n_actions  = int(env.action_space.n)

  def reset(self) -> int:
    obs, _info = self.env.reset()
    return int(obs)
  
  def step(self, action: int):
    obs, reward, terminated, truncated, info = self.env.step(action)
    done = bool(terminated or truncated)
    return int(obs), float(reward), done, info