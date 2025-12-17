import numpy as np
from rl.algorithms.base import AgentConfig
from rl.algorithms.sarsa import SARSAAgent


def test_sarsa_update_td_and_q_update():
    cfg = AgentConfig(n_states=3, n_actions=2, alpha=0.5, gamma=0.9, epsilon=0.0, seed=123)
    agent = SARSAAgent(cfg)
    # Set a small Q-table with known values
    agent.Q = np.array(
        [[0.0, 1.0],
         [2.0, 0.5],
         [0.25, -0.25]],
        dtype=np.float32,
    )

    s, a = 0, 1
    r = 0.5
    s_next, a_next = 1, 0
    done = False

    # SARSA target = r + gamma * Q(s', a')
    expected_target = r + cfg.gamma * float(agent.Q[s_next, a_next])
    expected_td = expected_target - float(agent.Q[s, a])
    expected_q = float(agent.Q[s, a]) + cfg.alpha * expected_td

    td = agent.update(s, a, r, s_next, a_next, done)

    assert np.isclose(td, expected_td)
    assert np.isclose(agent.Q[s, a], expected_q)


def test_sarsa_update_terminal_uses_immediate_reward():
    cfg = AgentConfig(n_states=2, n_actions=2, alpha=0.3, gamma=0.99, epsilon=0.0, seed=1)
    agent = SARSAAgent(cfg)
    agent.Q = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32)

    s, a = 1, 1
    r = 1.0
    s_next, a_next = 0, 0
    done = True

    expected_target = r  # terminal: no bootstrap
    expected_td = expected_target - float(agent.Q[s, a])
    expected_q = float(agent.Q[s, a]) + cfg.alpha * expected_td

    td = agent.update(s, a, r, s_next, a_next, done)

    assert np.isclose(td, expected_td)
    assert np.isclose(agent.Q[s, a], expected_q)
