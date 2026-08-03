"""
trainer.py

Double DQN Agent tailored for the Graph Neural Network architecture.
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from models.common.losses import huber_loss
from models.gnn.network import GNNDQNetwork
from models.gnn.replay_buffer import GraphReplayBuffer
from models.dqn.target_network import hard_update

try:
    from torch_geometric.data import Data, Batch
except ImportError:
    pass


class GNNDQNAgent:
    """Double DQN agent using Graph Neural Networks.

    Uses PyTorch Geometric Batch objects instead of standard tensors
    for state representations during the learn() step.
    """

    def __init__(self, config: dict):
        """Initialize the agent.

        Args:
            config: Full configuration dictionary containing 'agent', 'network',
                    and optionally 'environment' settings.
        """
        agent_cfg = config.get("agent", {})
        net_cfg = config.get("network", {})
        
        # Hyperparameters
        self.gamma = agent_cfg.get("gamma", 0.99)
        self.epsilon = agent_cfg.get("epsilon_start", 1.0)
        self.epsilon_decay = agent_cfg.get("epsilon_decay", 0.99)
        self.epsilon_min = agent_cfg.get("epsilon_min", 0.01)
        self.learning_rate = agent_cfg.get("learning_rate", 0.001)
        self.batch_size = agent_cfg.get("batch_size", 64)
        self.target_update_freq = agent_cfg.get("target_update_freq", 10)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dimensions
        self.action_size = 5  # Up, Down, Left, Right, Stay
        self.node_feature_dim = 9  # 8 CellTypes + 1 visit_count

        # GNN Architecture
        gcn_dims = net_cfg.get("gcn_hidden_dims", [64, 64, 64])
        mlp_dims = net_cfg.get("mlp_hidden_dims", [128, 64])

        # Networks
        self.q_network = GNNDQNetwork(
            node_feature_dim=self.node_feature_dim,
            action_size=self.action_size,
            gcn_hidden_dims=gcn_dims,
            mlp_hidden_dims=mlp_dims,
        ).to(self.device)

        self.target_network = GNNDQNetwork(
            node_feature_dim=self.node_feature_dim,
            action_size=self.action_size,
            gcn_hidden_dims=gcn_dims,
            mlp_hidden_dims=mlp_dims,
        ).to(self.device)

        hard_update(self.target_network, self.q_network)
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Replay Buffer
        buffer_size = agent_cfg.get("replay_buffer_size", 50000)
        self.memory = GraphReplayBuffer(capacity=buffer_size)

    def act(self, obs: Dict[str, np.ndarray], explore: bool = True) -> int:
        """Select an action using epsilon-greedy policy.

        Args:
            obs:     Graph observation dict (node_features, edge_index).
            explore: If True, use epsilon-greedy. If False, purely greedy.

        Returns:
            Integer action index.
        """
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)

        # Convert dict to PyG Data then Batch (size 1)
        data = Data(
            x=torch.FloatTensor(obs["node_features"]),
            edge_index=torch.LongTensor(obs["edge_index"])
        )
        batch = Batch.from_data_list([data]).to(self.device)

        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(batch)
        self.q_network.train()

        return int(torch.argmax(q_values).item())

    def learn(self) -> Optional[float]:
        """Perform one step of Double DQN gradient descent.

        Returns:
            The computed loss as a float, or None if buffer isn't ready.
        """
        if not self.memory.is_ready(self.batch_size):
            return None

        # 1. Sample from buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Move PyG Batches and tensors to device
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # 2. Compute current Q values: Q(s, a)
        # q_values shape: (batch_size, action_size)
        q_values = self.q_network(states)
        # current_q shape: (batch_size,)
        current_q = q_values.gather(1, actions).squeeze(1)

        # 3. Compute target Q values using Double DQN
        with torch.no_grad():
            # Action selection: argmax_a Q_main(s', a)
            next_q_main = self.q_network(next_states)
            best_next_actions = next_q_main.argmax(dim=1, keepdim=True)

            # Action evaluation: Q_target(s', best_a)
            next_q_target = self.target_network(next_states)
            next_max_q = next_q_target.gather(1, best_next_actions).squeeze(1)

            # Target = R + gamma * Q_target * (1 - done)
            target_q = rewards + self.gamma * next_max_q * (~dones)

        # 4. Compute loss
        loss = huber_loss(current_q, target_q)

        # 5. Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self) -> None:
        """Multiply epsilon by decay rate, bounded by minimum."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self) -> None:
        """Hard update the target network weights."""
        hard_update(self.target_network, self.q_network)

    def save_checkpoint(self, filepath: str) -> None:
        """Save network weights, optimizer, and epsilon."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load network weights, optimizer, and epsilon."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
