import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import random

print("Simulating GCN-DQN Training Loop and Replay Buffer...")

# 1. Custom Graph Replay Buffer
class GraphReplayBuffer:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state_graph, action, reward, next_state_graph, done):
        """Saves a PyG Data object instead of a numpy array!"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            
        # We store the transition as a dictionary
        self.memory[self.position] = {
            'state': state_graph,
            'action': action,
            'reward': reward,
            'next_state': next_state_graph,
            'done': done
        }
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# 2. Creating Dummy Graphs to simulate Environment Steps
def create_dummy_graph(is_agent_node=0):
    """Creates a 3-node graph. Node 0 is the agent's location if is_agent_node=0."""
    x = torch.zeros((3, 3)) # Features: [is_wall, is_exit, is_agent]
    x[is_agent_node, 2] = 1.0 # Set agent position
    
    edge_index = torch.tensor([
        [0, 1, 1, 2],
        [1, 0, 2, 1]
    ], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index)

buffer = GraphReplayBuffer(100)

# Simulate 10 steps in an environment
for step in range(10):
    state = create_dummy_graph(is_agent_node=step % 3)
    action = random.randint(0, 3)
    reward = 1.0 if step == 9 else -0.1
    next_state = create_dummy_graph(is_agent_node=(step + 1) % 3)
    done = step == 9
    
    buffer.push(state, action, reward, next_state, done)

print(f"Stored {len(buffer)} graph transitions in Replay Buffer.")

# 3. Sampling and Batching
batch = buffer.sample(4)

# We use PyTorch Geometric's DataLoader to combine multiple disconnected graphs into one giant Batch
state_graphs = [transition['state'] for transition in batch]
loader = DataLoader(state_graphs, batch_size=4)
batched_states = next(iter(loader))

print("\n--- Batched State Graphs ---")
print(f"Combined Nodes: {batched_states.num_nodes}") # 4 graphs * 3 nodes = 12 nodes
print(f"Combined Edges: {batched_states.num_edges}") # 4 graphs * 4 edges = 16 edges
print(f"Batch Vector mapping nodes to graphs:\n{batched_states.batch}")
print("\nThis batched object is what we feed into our GNNDQNetwork!")
