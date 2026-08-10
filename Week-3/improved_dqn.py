import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections
import matplotlib.pyplot as plt

# 1. Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
        
    def __len__(self):
        return len(self.buffer)

# 2. DQN Architecture
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

# Training Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LR = 1e-3
TARGET_UPDATE = 10  # How often to update the target network
MAX_EPISODES = 300

env = gym.make('CartPole-v1')
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# 3. Two Networks: Policy and Target
policy_net = DQN(n_states, n_actions)
target_net = DQN(n_states, n_actions)
target_net.load_state_dict(policy_net.state_dict()) # Clone weights
target_net.eval() # Target network is frozen

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(10000)
epsilon = EPSILON_START
rewards_history = []

print("Training Improved DQN with Experience Replay & Target Network...")

for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Epsilon-greedy action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            action = q_values.argmax().item()
            
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store in Replay Buffer
        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        # Train from Replay Buffer
        if len(memory) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
            
            states_t = torch.FloatTensor(states)
            actions_t = torch.LongTensor(actions).unsqueeze(1)
            rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
            next_states_t = torch.FloatTensor(next_states)
            dones_t = torch.FloatTensor(dones).unsqueeze(1)
            
            # Current Q-Values
            q_values = policy_net(states_t).gather(1, actions_t)
            
            # Next Q-Values using TARGET NETWORK
            with torch.no_grad():
                max_next_q_values = target_net(next_states_t).max(1)[0].unsqueeze(1)
                expected_q_values = rewards_t + (GAMMA * max_next_q_values * (1 - dones_t))
                
            loss = nn.MSELoss()(q_values, expected_q_values)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # Update Target Network
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    rewards_history.append(total_reward)
    
    avg_reward = np.mean(rewards_history[-50:])
    if episode % 10 == 0:
        print(f"Episode {episode} | Reward: {total_reward} | Avg(50): {avg_reward:.1f} | Epsilon: {epsilon:.2f}")
        
    if avg_reward >= 475:
        print(f"✅ Environment solved in {episode} episodes!")
        break

env.close()

# Plot Results
plt.figure(figsize=(10,5))
plt.plot(rewards_history, alpha=0.6, label="Episode Reward")
# Rolling Average
rolling_avg = [np.mean(rewards_history[max(0, i-50):i+1]) for i in range(len(rewards_history))]
plt.plot(rolling_avg, color='orange', label="50-Episode Avg", linewidth=2)
plt.axhline(y=475, color='r', linestyle='--', label="Solved Threshold")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Improved DQN Performance (CartPole-v1)")
plt.legend()
plt.savefig("improved_dqn_results.png")
print("Plot saved to improved_dqn_results.png")
