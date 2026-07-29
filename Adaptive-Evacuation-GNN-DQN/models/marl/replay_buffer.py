import random
import numpy as np
import torch
from collections import deque
from torch_geometric.data import Data, Batch

class MARLReplayBuffer:
    """
    Experience replay buffer for Multi-Agent GNN training.
    Stores full graph states, but associates them with specific agents
    and their unique node IDs, actions, rewards, and terminations.
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        agent_node_ids: list[int],
        actions: list[int],
        rewards: list[float],
        next_node_features: np.ndarray,
        next_agent_node_ids: list[int],
        dones: list[bool]
    ):
        """
        Stores a single MARL timestep.
        
        Because agents can finish at different times, we store the data 
        for all agents that were ACTIVE at the START of this timestep.
        """
        experience = (
            node_features,
            edge_index,
            agent_node_ids,
            actions,
            rewards,
            next_node_features,
            next_agent_node_ids,
            dones
        )
        self.buffer.append(experience)

    def sample(self, batch_size: int, device: torch.device):
        """
        Samples a batch of MARL experiences.
        Uses PyTorch Geometric's `Batch.from_data_list` to handle graph batching.
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        state_graphs = []
        next_state_graphs = []
        
        all_agent_node_ids = []
        all_actions = []
        all_rewards = []
        all_next_agent_node_ids = []
        all_dones = []
        
        # We must adjust the agent_node_ids because Batching merges multiple graphs
        # into one giant disconnected graph. Node 0 of graph 1 becomes Node 0+N.
        current_node_offset = 0
        
        for exp in batch:
            nf, ei, a_ids, acts, rews, next_nf, next_a_ids, dones = exp
            
            # Create PyG Data objects
            state_data = Data(
                x=torch.FloatTensor(nf),
                edge_index=torch.LongTensor(ei)
            )
            state_graphs.append(state_data)
            
            next_state_data = Data(
                x=torch.FloatTensor(next_nf),
                edge_index=torch.LongTensor(ei) # Structure doesn't change
            )
            next_state_graphs.append(next_state_data)
            
            # Adjust node IDs by the cumulative node count
            num_nodes = nf.shape[0]
            
            for i in range(len(a_ids)):
                all_agent_node_ids.append(a_ids[i] + current_node_offset)
                all_next_agent_node_ids.append(next_a_ids[i] + current_node_offset)
                all_actions.append(acts[i])
                all_rewards.append(rews[i])
                all_dones.append(dones[i])
                
            current_node_offset += num_nodes

        # Batch the graphs
        batch_state = Batch.from_data_list(state_graphs).to(device)
        batch_next_state = Batch.from_data_list(next_state_graphs).to(device)
        
        # Convert to tensors
        t_agent_node_ids = torch.LongTensor(all_agent_node_ids).to(device)
        t_actions = torch.LongTensor(all_actions).unsqueeze(1).to(device)
        t_rewards = torch.FloatTensor(all_rewards).unsqueeze(1).to(device)
        t_next_agent_node_ids = torch.LongTensor(all_next_agent_node_ids).to(device)
        t_dones = torch.FloatTensor(all_dones).unsqueeze(1).to(device)

        return (
            batch_state, 
            t_agent_node_ids, 
            t_actions, 
            t_rewards, 
            batch_next_state, 
            t_next_agent_node_ids, 
            t_dones
        )

    def __len__(self) -> int:
        return len(self.buffer)
