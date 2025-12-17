
from rl.algorithms.base import AgentConfig
from rl.algorithms.sarsa import SARSAAgent


def test_epsilon_zero_is_greedy():
    cfg = AgentConfig(n_states=1, n_actions=2, alpha=0.1, gamma=0.9, epsilon=0.0, seed=10)
    agent = SARSAAgent(cfg)
    # Make action 1 clearly better
    agent.Q[0, 0] = 0.0
    agent.Q[0, 1] = 5.0

    actions = [agent.act(0, policy="epsilon_greedy") for _ in range(50)]
    assert set(actions) == {1}


def test_epsilon_one_explores_both_actions():
    cfg = AgentConfig(n_states=1, n_actions=2, alpha=0.1, gamma=0.9, epsilon=1.0, seed=123)
    agent = SARSAAgent(cfg)
    agent.Q[0, 0] = 0.0
    agent.Q[0, 1] = 5.0

    actions = [agent.act(0, policy="epsilon_greedy") for _ in range(200)]
    # With full exploration and fixed RNG seed, we should see both 0 and 1
    assert 0 in actions and 1 in actions


def test_policy_greedy_overrides_epsilon():
    cfg = AgentConfig(n_states=1, n_actions=2, alpha=0.1, gamma=0.9, epsilon=1.0, seed=5)
    agent = SARSAAgent(cfg)
    agent.Q[0, 0] = 1.0
    agent.Q[0, 1] = 0.0

    actions = [agent.act(0, policy="greedy") for _ in range(20)]
    assert set(actions) == {0}
