"""
trainer.py

DQN Agent with Double DQN support, epsilon-greedy exploration,
experience replay, target network, and checkpoint persistence.
"""

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.optim as optim

from models.common.layers import one_hot_encode_grid, batch_one_hot_encode
from models.common.losses import huber_loss
from models.dqn.network import DQNetwork
from models.dqn.replay_buffer import ReplayBuffer
from models.dqn.target_network import hard_update


class DQNAgent:
    """Deep Q-Network agent with Double DQN.

    Double DQN decouples action selection from value estimation:
      - The Q-network selects the best action for the next state.
      - The target network estimates the value of that action.
    This reduces overestimation bias that plagues standard DQN.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Args:
            config: Configuration dict with 'agent', 'network', and 'training'
                    sections (typically loaded from configs/dqn.yaml).
        """
        agent_cfg = config.get("agent", {})
        net_cfg = config.get("network", {})
        train_cfg = config.get("training", {})

        # --- Hyperparameters ---
        self.gamma = agent_cfg.get("gamma", 0.99)
        self.epsilon = agent_cfg.get("epsilon_start", 1.0)
        self.epsilon_decay = agent_cfg.get("epsilon_decay", 0.997)
        self.epsilon_min = agent_cfg.get("epsilon_min", 0.01)
        self.learning_rate = agent_cfg.get("learning_rate", 0.001)
        self.batch_size = agent_cfg.get("batch_size", 64)
        self.target_update_freq = agent_cfg.get("target_update_freq", 10)

        # --- Environment dimensions ---
        # Default: 10×10 grid, 8 cell types, 5 actions
        grid_rows = config.get("environment", {}).get("grid_rows", 10)
        grid_cols = config.get("environment", {}).get("grid_cols", 10)
        num_cell_types = 8
        self.state_size = grid_rows * grid_cols
        self.input_size = self.state_size * num_cell_types  # one-hot: 800
        self.action_size = 5
        self.num_cell_types = num_cell_types

        # --- Networks ---
        hidden_layers = net_cfg.get("hidden_layers", [256, 256, 128])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = DQNetwork(
            input_size=self.input_size,
            action_size=self.action_size,
            hidden_layers=hidden_layers,
        ).to(self.device)

        self.target_network = DQNetwork(
            input_size=self.input_size,
            action_size=self.action_size,
            hidden_layers=hidden_layers,
        ).to(self.device)

        hard_update(self.target_network, self.q_network)

        # --- Optimizer ---
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=self.learning_rate
        )

        # --- Replay buffer ---
        buffer_size = agent_cfg.get("replay_buffer_size", 50_000)
        self.memory = ReplayBuffer(capacity=buffer_size)

        # --- Seed ---
        seed = train_cfg.get("seed", None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """Select an action using epsilon-greedy policy.

        Args:
            state:   Raw observation from the environment (integer array).
            explore: If False, always exploit (greedy). Used during evaluation.

        Returns:
            Integer action id (0–4).
        """
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state_tensor = one_hot_encode_grid(state, self.num_cell_types)
        state_tensor = state_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return q_values.argmax(dim=1).item()

    # ------------------------------------------------------------------
    # Learning (Double DQN)
    # ------------------------------------------------------------------
    def learn(self) -> Optional[float]:
        """Sample a batch and perform one gradient step (Double DQN).

        Double DQN update rule:
            a* = argmax_a Q_online(s', a)
            y  = r + γ * Q_target(s', a*) * (1 - done)
            loss = Huber(Q_online(s, a) - y)

        Returns:
            Loss value as float, or None if buffer is not ready.
        """
        if not self.memory.is_ready(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # --- One-hot encode ---
        states_t = batch_one_hot_encode(states, self.num_cell_types).to(self.device)
        next_states_t = batch_one_hot_encode(next_states, self.num_cell_types).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # --- Current Q-values ---
        current_q = self.q_network(states_t).gather(
            1, actions_t.unsqueeze(1)
        ).squeeze(1)

        # --- Double DQN targets ---
        with torch.no_grad():
            # Q-network selects the best action
            best_actions = self.q_network(next_states_t).argmax(dim=1)
            # Target network evaluates the selected action
            next_q = self.target_network(next_states_t).gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        # --- Compute loss and update ---
        loss = huber_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Target network update
    # ------------------------------------------------------------------
    def update_target(self) -> None:
        """Copy Q-network weights to target network."""
        hard_update(self.target_network, self.q_network)

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------
    def decay_epsilon(self) -> None:
        """Multiplicative epsilon decay with minimum floor."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str) -> None:
        """Save model weights and training state to disk.

        Args:
            path: File path for the checkpoint (.pt file).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "input_size": self.input_size,
            "action_size": self.action_size,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model weights and training state from disk.

        Args:
            path: File path to the checkpoint (.pt file).
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
