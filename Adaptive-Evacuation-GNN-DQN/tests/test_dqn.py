"""
test_dqn.py

Unit tests for Phase 2 — DQN agent, network, replay buffer,
common utilities, metrics, and environment integration.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.common.layers import one_hot_encode_grid, batch_one_hot_encode
from models.common.losses import huber_loss
from models.dqn.network import DQNetwork
from models.dqn.replay_buffer import ReplayBuffer
from models.dqn.target_network import hard_update, soft_update
from models.dqn.trainer import DQNAgent
from evaluation.metrics import TrainingMetrics


# ──────────────────────────────────────────────────────────────────────
# Test Configuration (minimal — fast tests)
# ──────────────────────────────────────────────────────────────────────
def _minimal_config():
    """Return a minimal DQN config for testing."""
    return {
        "environment": {"grid_rows": 10, "grid_cols": 10, "config_path": "configs/default.yaml"},
        "agent": {
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_decay": 0.99,
            "epsilon_min": 0.01,
            "learning_rate": 0.001,
            "batch_size": 4,  # Small batch for fast tests
            "replay_buffer_size": 100,
            "target_update_freq": 5,
        },
        "network": {"hidden_layers": [32, 16]},  # Tiny network for speed
        "training": {"seed": 42},
    }


# ──────────────────────────────────────────────────────────────────────
# Test: One-Hot Encoding
# ──────────────────────────────────────────────────────────────────────
class TestOneHotEncoding(unittest.TestCase):

    def test_single_observation_shape(self):
        """one_hot_encode_grid returns correct shape."""
        obs = np.zeros(100, dtype=np.int32)
        result = one_hot_encode_grid(obs, num_classes=8)
        self.assertEqual(result.shape, (800,))
        self.assertEqual(result.dtype, torch.float32)

    def test_single_observation_values(self):
        """One-hot encoding produces correct values."""
        obs = np.array([0, 1, 2, 6], dtype=np.int32)
        result = one_hot_encode_grid(obs, num_classes=8)
        # Cell 0 (EMPTY=0): one-hot at index 0
        self.assertAlmostEqual(result[0].item(), 1.0)
        self.assertAlmostEqual(result[1].item(), 0.0)
        # Cell 1 (WALL=1): one-hot at index 9 (1*8 + 1)
        self.assertAlmostEqual(result[9].item(), 1.0)
        # Cell 2 (AGENT=2): one-hot at index 18 (2*8 + 2)
        self.assertAlmostEqual(result[18].item(), 1.0)

    def test_batch_encoding_shape(self):
        """batch_one_hot_encode returns correct batch shape."""
        batch = np.zeros((4, 100), dtype=np.int32)
        result = batch_one_hot_encode(batch, num_classes=8)
        self.assertEqual(result.shape, (4, 800))

    def test_batch_encoding_sum(self):
        """Each cell should have exactly one 1 in its one-hot slice."""
        obs = np.random.randint(0, 8, size=(1, 100), dtype=np.int32)
        result = batch_one_hot_encode(obs, num_classes=8)
        # Reshape to (100, 8) and check each row sums to 1
        reshaped = result.reshape(100, 8)
        row_sums = reshaped.sum(dim=1)
        for i in range(100):
            self.assertAlmostEqual(row_sums[i].item(), 1.0)


# ──────────────────────────────────────────────────────────────────────
# Test: Huber Loss
# ──────────────────────────────────────────────────────────────────────
class TestHuberLoss(unittest.TestCase):

    def test_zero_loss(self):
        """Loss is zero when predicted == target."""
        predicted = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        loss = huber_loss(predicted, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_positive_loss(self):
        """Loss is positive when predicted != target."""
        predicted = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 3.0, 4.0])
        loss = huber_loss(predicted, target)
        self.assertGreater(loss.item(), 0.0)

    def test_symmetric(self):
        """Loss is symmetric: L(a, b) == L(b, a)."""
        a = torch.tensor([1.0, 5.0])
        b = torch.tensor([3.0, 2.0])
        self.assertAlmostEqual(huber_loss(a, b).item(), huber_loss(b, a).item(), places=5)


# ──────────────────────────────────────────────────────────────────────
# Test: DQNetwork
# ──────────────────────────────────────────────────────────────────────
class TestDQNetwork(unittest.TestCase):

    def test_forward_pass_shape(self):
        """Network produces correct output shape."""
        net = DQNetwork(input_size=800, action_size=5, hidden_layers=[64, 32])
        x = torch.randn(8, 800)  # batch of 8
        out = net(x)
        self.assertEqual(out.shape, (8, 5))

    def test_default_architecture(self):
        """Default hidden layers are [256, 256, 128]."""
        net = DQNetwork()
        # Count linear layers
        linear_layers = [m for m in net.network if isinstance(m, torch.nn.Linear)]
        self.assertEqual(len(linear_layers), 4)  # 3 hidden + 1 output
        self.assertEqual(linear_layers[0].in_features, 800)
        self.assertEqual(linear_layers[0].out_features, 256)
        self.assertEqual(linear_layers[-1].out_features, 5)

    def test_gradient_flow(self):
        """Gradients flow through the network."""
        net = DQNetwork(input_size=16, action_size=3, hidden_layers=[8])
        x = torch.randn(2, 16, requires_grad=True)
        out = net(x)
        loss = out.sum()
        loss.backward()
        # Check that all parameters have gradients
        for param in net.parameters():
            self.assertIsNotNone(param.grad)


# ──────────────────────────────────────────────────────────────────────
# Test: Replay Buffer
# ──────────────────────────────────────────────────────────────────────
class TestReplayBuffer(unittest.TestCase):

    def test_push_and_len(self):
        """Buffer correctly tracks size."""
        buf = ReplayBuffer(capacity=10)
        self.assertEqual(len(buf), 0)
        buf.push(np.zeros(4), 0, 1.0, np.zeros(4), False)
        self.assertEqual(len(buf), 1)

    def test_capacity_overflow(self):
        """Buffer discards oldest when full."""
        buf = ReplayBuffer(capacity=3)
        for i in range(5):
            buf.push(np.array([i]), 0, 1.0, np.array([i+1]), False)
        self.assertEqual(len(buf), 3)

    def test_sample_shape(self):
        """Sample returns correctly shaped arrays."""
        buf = ReplayBuffer(capacity=100)
        for i in range(10):
            buf.push(np.zeros(100, dtype=np.int32), 0, 1.0, np.zeros(100, dtype=np.int32), False)
        states, actions, rewards, next_states, dones = buf.sample(4)
        self.assertEqual(states.shape, (4, 100))
        self.assertEqual(actions.shape, (4,))
        self.assertEqual(rewards.shape, (4,))
        self.assertEqual(next_states.shape, (4, 100))
        self.assertEqual(dones.shape, (4,))

    def test_is_ready(self):
        """is_ready reports correctly."""
        buf = ReplayBuffer(capacity=100)
        self.assertFalse(buf.is_ready(5))
        for i in range(5):
            buf.push(np.zeros(4), 0, 1.0, np.zeros(4), False)
        self.assertTrue(buf.is_ready(5))


# ──────────────────────────────────────────────────────────────────────
# Test: Target Network
# ──────────────────────────────────────────────────────────────────────
class TestTargetNetwork(unittest.TestCase):

    def test_hard_update_copies_exactly(self):
        """hard_update makes target == source."""
        source = DQNetwork(input_size=16, action_size=3, hidden_layers=[8])
        target = DQNetwork(input_size=16, action_size=3, hidden_layers=[8])
        hard_update(target, source)
        for sp, tp in zip(source.parameters(), target.parameters()):
            self.assertTrue(torch.equal(sp, tp))

    def test_soft_update_blends(self):
        """soft_update with tau=1.0 is equivalent to hard_update."""
        source = DQNetwork(input_size=16, action_size=3, hidden_layers=[8])
        target = DQNetwork(input_size=16, action_size=3, hidden_layers=[8])
        soft_update(target, source, tau=1.0)
        for sp, tp in zip(source.parameters(), target.parameters()):
            self.assertTrue(torch.allclose(sp, tp))

    def test_soft_update_tau_zero_no_change(self):
        """soft_update with tau=0.0 leaves target unchanged."""
        source = DQNetwork(input_size=16, action_size=3, hidden_layers=[8])
        target = DQNetwork(input_size=16, action_size=3, hidden_layers=[8])
        # Store original target params
        original_params = [p.clone() for p in target.parameters()]
        soft_update(target, source, tau=0.0)
        for orig, current in zip(original_params, target.parameters()):
            self.assertTrue(torch.equal(orig, current))


# ──────────────────────────────────────────────────────────────────────
# Test: DQNAgent
# ──────────────────────────────────────────────────────────────────────
class TestDQNAgent(unittest.TestCase):

    def setUp(self):
        self.config = _minimal_config()
        self.agent = DQNAgent(self.config)

    def test_act_returns_valid_action(self):
        """Agent returns an action in [0, 4]."""
        obs = np.zeros(100, dtype=np.int32)
        for _ in range(20):
            action = self.agent.act(obs, explore=True)
            self.assertIn(action, range(5))

    def test_act_greedy(self):
        """Greedy action is deterministic for same input."""
        obs = np.zeros(100, dtype=np.int32)
        self.agent.epsilon = 0.0  # No exploration
        action1 = self.agent.act(obs, explore=True)
        action2 = self.agent.act(obs, explore=True)
        self.assertEqual(action1, action2)

    def test_learn_returns_none_when_buffer_empty(self):
        """learn() returns None when buffer has fewer samples than batch_size."""
        loss = self.agent.learn()
        self.assertIsNone(loss)

    def test_learn_returns_loss(self):
        """learn() returns a float loss when buffer is ready."""
        obs = np.zeros(100, dtype=np.int32)
        # Fill buffer beyond batch_size
        for _ in range(10):
            self.agent.memory.push(obs, 0, 1.0, obs, False)
        loss = self.agent.learn()
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)

    def test_epsilon_decay(self):
        """decay_epsilon reduces epsilon."""
        initial = self.agent.epsilon
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, initial)

    def test_epsilon_floor(self):
        """Epsilon never goes below epsilon_min."""
        self.agent.epsilon = 0.011
        for _ in range(1000):
            self.agent.decay_epsilon()
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_min)

    def test_checkpoint_save_load(self):
        """Checkpoint round-trips correctly."""
        obs = np.zeros(100, dtype=np.int32)
        self.agent.epsilon = 0.0
        action_before = self.agent.act(obs, explore=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_checkpoint.pt")
            self.agent.save_checkpoint(path)

            # Create a new agent and load
            new_agent = DQNAgent(self.config)
            new_agent.load_checkpoint(path)
            new_agent.epsilon = 0.0
            action_after = new_agent.act(obs, explore=False)

        self.assertEqual(action_before, action_after)


# ──────────────────────────────────────────────────────────────────────
# Test: Training Metrics
# ──────────────────────────────────────────────────────────────────────
class TestTrainingMetrics(unittest.TestCase):

    def test_record_and_length(self):
        """Recording episodes increments the lists."""
        m = TrainingMetrics()
        m.record(reward=10.0, steps=50, success=True, loss=0.5, epsilon=0.9)
        self.assertEqual(len(m.rewards), 1)

    def test_rolling_average(self):
        """Rolling average computes correctly."""
        m = TrainingMetrics()
        for i in range(10):
            m.record(reward=float(i), steps=i, success=(i > 5), loss=0.1, epsilon=0.5)
        rolling = m.rolling_average(window=5)
        # Last 5 rewards: 5, 6, 7, 8, 9 → avg = 7.0
        self.assertAlmostEqual(rolling["avg_reward"], 7.0)
        # Last 5 successes: True, True, True, True, False → 80%
        self.assertAlmostEqual(rolling["success_rate"], 0.8)

    def test_to_dict(self):
        """to_dict returns all expected keys."""
        m = TrainingMetrics()
        m.record(reward=1.0, steps=10, success=False, loss=0.1, epsilon=0.5)
        d = m.to_dict()
        expected_keys = {"reward", "steps", "success", "loss", "epsilon", "fire_count", "reason"}
        self.assertEqual(set(d.keys()), expected_keys)

    def test_csv_round_trip(self):
        """Metrics survive save/load to CSV."""
        m = TrainingMetrics()
        m.record(reward=42.5, steps=100, success=True, loss=0.123, epsilon=0.5, fire_count=3, reason="reached_exit")
        m.record(reward=-20.0, steps=50, success=False, loss=0.456, epsilon=0.4, fire_count=10, reason="hit_fire")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_metrics.csv")
            m.save_csv(path)
            loaded = TrainingMetrics.load_csv(path)

        self.assertEqual(len(loaded.rewards), 2)
        self.assertAlmostEqual(loaded.rewards[0], 42.5, places=1)
        self.assertEqual(loaded.reasons[1], "hit_fire")


# ──────────────────────────────────────────────────────────────────────
# Test: Integration (Agent + Environment)
# ──────────────────────────────────────────────────────────────────────
class TestIntegration(unittest.TestCase):

    def test_agent_env_interaction(self):
        """Agent can interact with the environment for 10 steps."""
        from utils.config_loader import load_config
        from environment.evacuation_env import EvacuationEnv

        env_config = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
        env = EvacuationEnv(env_config)

        config = _minimal_config()
        agent = DQNAgent(config)

        obs, info = env.reset(seed=42)
        total_reward = 0.0

        for step in range(10):
            action = agent.act(obs, explore=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.memory.push(obs, action, reward, next_obs, terminated or truncated)
            loss = agent.learn()
            obs = next_obs
            total_reward += reward
            if terminated or truncated:
                break

        # Should complete without errors
        self.assertIsInstance(total_reward, float)
        env.close()

    def test_full_episode_completes(self):
        """A full episode with DQN agent completes (terminated or truncated)."""
        from utils.config_loader import load_config
        from environment.evacuation_env import EvacuationEnv

        env_config = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
        env = EvacuationEnv(env_config)

        config = _minimal_config()
        agent = DQNAgent(config)

        obs, _ = env.reset(seed=123)
        done = False
        steps = 0

        while not done and steps < 200:
            action = agent.act(obs, explore=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        self.assertTrue(done)
        self.assertIn(info["reason"], [
            "reached_exit", "hit_fire", "fire_caught_agent", "max_steps_exceeded"
        ])
        env.close()


if __name__ == "__main__":
    unittest.main()
