import gymnasium as gym
from rl.adapter.frozenlake import FrozenLakeAdapter
from rl.algorithms.base import AgentConfig
from rl.algorithms.q_learning import QLearningAgent


def rollout(env_adapter: FrozenLakeAdapter, agent: QLearningAgent, max_steps: int = 200):
    s = env_adapter.reset(seed=0)
    traj = []
    for _ in range(max_steps):
        a = agent.act(s, policy="epsilon_greedy")
        s_next, r, done, _info = env_adapter.step(a)
        traj.append((s, a, r))
        if done:
            break
        s = s_next
    return traj


def test_reproducible_rollout_same_seed():
    # Deterministic env variant + fixed agent RNG
    env1 = FrozenLakeAdapter(gym.make("FrozenLake-v1", is_slippery=False))
    env2 = FrozenLakeAdapter(gym.make("FrozenLake-v1", is_slippery=False))

    cfg = AgentConfig(n_states=env1.n_states, n_actions=env1.n_actions, epsilon=0.3, seed=123)
    agent1 = QLearningAgent(cfg)
    agent2 = QLearningAgent(cfg)

    traj1 = rollout(env1, agent1)
    traj2 = rollout(env2, agent2)

    assert traj1 == traj2
