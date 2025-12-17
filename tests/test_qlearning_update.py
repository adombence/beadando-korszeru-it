import numpy as np
from rl.algorithms.base import AgentConfig
from rl.algorithms.q_learning import QLearningAgent


def test_q_learning_update_td_and_q_update():
    cfg = AgentConfig(n_states=3, n_actions=2, alpha=0.4, gamma=0.95, epsilon=0.0, seed=42)
    agent = QLearningAgent(cfg)
    agent.Q = np.array(
        [[0.0, 0.5],
         [1.0, 2.0],
         [0.25, -0.25]],
        dtype=np.float32,
    )

    s, a = 0, 1
    r = 0.2
    s_next = 1
    a_next = 0  # ignored in Q-learning update
    done = False

    # Q-learning target = r + gamma * max_a' Q(s', a')
    max_next = float(agent.Q[s_next].max())
    expected_target = r + cfg.gamma * max_next
    expected_td = expected_target - float(agent.Q[s, a])
    expected_q = float(agent.Q[s, a]) + cfg.alpha * expected_td

    td = agent.update(s, a, r, s_next, a_next, done)

    assert np.isclose(td, expected_td)
    assert np.isclose(agent.Q[s, a], expected_q)


def test_q_learning_update_terminal_uses_immediate_reward():
    cfg = AgentConfig(n_states=2, n_actions=2, alpha=0.25, gamma=0.9, epsilon=0.0, seed=7)
    agent = QLearningAgent(cfg)
    agent.Q = np.array([[0.5, 0.0], [1.0, 2.0]], dtype=np.float32)

    s, a = 1, 0
    r = 1.5
    s_next, a_next = 0, 0
    done = True

    expected_target = r
    expected_td = expected_target - float(agent.Q[s, a])
    expected_q = float(agent.Q[s, a]) + cfg.alpha * expected_td

    td = agent.update(s, a, r, s_next, a_next, done)

    assert np.isclose(td, expected_td)
    assert np.isclose(agent.Q[s, a], expected_q)
