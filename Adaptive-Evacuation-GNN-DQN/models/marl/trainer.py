import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Dict, Any, List

from models.marl.network import MARL_GNNDQNetwork
from models.marl.replay_buffer import MARLReplayBuffer

class MARLAgent:
    def __init__(self, config: Dict[str, Any]):
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        
        self.input_dim = config["agent"]["input_dim"]
        self.hidden_dim = config["agent"]["hidden_dim"]
        self.num_actions = config["agent"]["num_actions"]
        
        self.lr = config["agent"]["learning_rate"]
        self.gamma = config["agent"]["gamma"]
        self.tau = config["agent"]["tau"]
        
        # Networks
        self.q_net = MARL_GNNDQNetwork(self.input_dim, self.hidden_dim, self.num_actions).to(self.device)
        self.target_net = MARL_GNNDQNetwork(self.input_dim, self.hidden_dim, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        # Buffer
        self.batch_size = config["training"]["batch_size"]
        self.buffer = MARLReplayBuffer(config["training"]["buffer_size"])
        
        # Exploration
        self.epsilon = config["exploration"]["epsilon_start"]
        self.epsilon_end = config["exploration"]["epsilon_end"]
        self.epsilon_decay = config["exploration"]["epsilon_decay"]

    def act(self, obs: Dict[str, np.ndarray], active_agents: List[bool], agent_node_ids: List[int], explore: bool = True) -> List[int]:
        """Returns a list of actions for ALL agents. Inactive agents get 0 (UP/STAY doesn't matter)."""
        actions = [4] * len(active_agents) # Default STAY
        
        # Get active agents only
        active_indices = [i for i, active in enumerate(active_agents) if active]
        if not active_indices:
            return actions
            
        if explore and random.random() < self.epsilon:
            for i in active_indices:
                actions[i] = random.randrange(self.num_actions)
            return actions
            
        with torch.no_grad():
            x = torch.FloatTensor(obs["node_features"]).to(self.device)
            edge_index = torch.LongTensor(obs["edge_index"]).to(self.device)
            
            # Only query active node IDs
            active_node_ids = [agent_node_ids[i] for i in active_indices]
            t_node_ids = torch.LongTensor(active_node_ids).to(self.device)
            
            # Forward pass: outputs shape (num_active_agents, num_actions)
            q_vals = self.q_net(x, edge_index, t_node_ids)
            best_actions = q_vals.argmax(dim=1).cpu().tolist()
            
            # Map back to full action list
            for idx, act in zip(active_indices, best_actions):
                actions[idx] = act
                
        return actions

    def step(self, obs, next_obs, active_agents, agent_node_ids, actions, rewards, next_agent_node_ids, dones):
        """Stores experience and triggers learning."""
        # Only store experience for agents that were active AT THE START of the step
        active_indices = [i for i, active in enumerate(active_agents) if active]
        if not active_indices:
            return
            
        f_agent_node_ids = [agent_node_ids[i] for i in active_indices]
        f_actions = [actions[i] for i in active_indices]
        f_rewards = [rewards[i] for i in active_indices]
        
        # For next node IDs, if they died this step, they don't have a next position.
        # But we must provide something for the tensor. We use the current position,
        # since their 'done' flag is True anyway, the Q-value will be 0.
        f_next_agent_node_ids = []
        for i in active_indices:
            n_id = next_agent_node_ids[i]
            if n_id == -1:
                f_next_agent_node_ids.append(agent_node_ids[i])
            else:
                f_next_agent_node_ids.append(n_id)
                
        f_dones = [dones[i] for i in active_indices]
        
        self.buffer.push(
            obs["node_features"], obs["edge_index"],
            f_agent_node_ids, f_actions, f_rewards,
            next_obs["node_features"], f_next_agent_node_ids, f_dones
        )
        
        if len(self.buffer) >= self.batch_size:
            self._learn()
            
    def _learn(self):
        batch = self.buffer.sample(self.batch_size, self.device)
        batch_state, t_agent_node_ids, t_actions, t_rewards, batch_next_state, t_next_agent_node_ids, t_dones = batch
        
        # Current Q
        current_q = self.q_net(batch_state.x, batch_state.edge_index, t_agent_node_ids).gather(1, t_actions)
        
        # Target Q
        with torch.no_grad():
            next_q = self.target_net(batch_next_state.x, batch_next_state.edge_index, t_next_agent_node_ids).max(1)[0].unsqueeze(1)
            target_q = t_rewards + (self.gamma * next_q * (1 - t_dones))
            
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
