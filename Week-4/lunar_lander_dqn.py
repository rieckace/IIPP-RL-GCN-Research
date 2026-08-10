import os
import sys
import subprocess

if sys.version_info >= (3, 14):
    fallback_python = r"C:\Users\Acer\AppData\Local\Programs\Python\Python312\python.exe"
    if os.path.exists(fallback_python):
        result = subprocess.run([fallback_python, os.path.abspath(__file__), *sys.argv[1:]])
        raise SystemExit(result.returncode)

    raise SystemExit(
        "LunarLander needs a Python interpreter with Box2D installed. "
        "Python 3.12 worked in this workspace, but the fallback interpreter was not found at: "
        f"{fallback_python}"
    )

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections
import matplotlib.pyplot as plt

# 1. Experience Replay Buffer (Larger for LunarLander)
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = collections.deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
        
    def __len__(self):
        return len(self.buffer)

# 2. DQN Architecture (Larger for LunarLander)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

# Training Hyperparameters Tuned for LunarLander
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.99  # Faster decay since episodes are longer
LR = 5e-4             # Smaller LR for stability
TARGET_UPDATE = 10
MAX_EPISODES = int(os.environ.get("LUNAR_LANDER_MAX_EPISODES", "600"))    # Needs more episodes to learn

# Note: Requires pip install gymnasium[box2d]
env = gym.make('LunarLander-v3')
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net = DQN(n_states, n_actions)
target_net = DQN(n_states, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(100000)
epsilon = EPSILON_START
rewards_history = []

print("Training DQN on LunarLander-v3...")

for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            action = q_values.argmax().item()
            
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        if len(memory) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
            
            states_t = torch.FloatTensor(states)
            actions_t = torch.LongTensor(actions).unsqueeze(1)
            rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
            next_states_t = torch.FloatTensor(next_states)
            dones_t = torch.FloatTensor(dones).unsqueeze(1)
            
            q_values = policy_net(states_t).gather(1, actions_t)
            
            with torch.no_grad():
                max_next_q_values = target_net(next_states_t).max(1)[0].unsqueeze(1)
                expected_q_values = rewards_t + (GAMMA * max_next_q_values * (1 - dones_t))
                
            loss = nn.MSELoss()(q_values, expected_q_values)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    rewards_history.append(total_reward)
    
    avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
    
    if episode % 10 == 0:
        print(f"Episode {episode} | Reward: {total_reward:.1f} | Avg(100): {avg_reward:.1f} | Epsilon: {epsilon:.2f}")
        
    if avg_reward >= 200.0:
        print(f"✅ Environment solved in {episode} episodes!")
        break

env.close()

plt.figure(figsize=(10,5))
plt.plot(rewards_history, alpha=0.6, label="Episode Reward")
rolling_avg = [np.mean(rewards_history[max(0, i-100):i+1]) for i in range(len(rewards_history))]
plt.plot(rolling_avg, color='orange', label="100-Episode Avg", linewidth=2)
plt.axhline(y=200, color='r', linestyle='--', label="Solved Threshold")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Performance (LunarLander-v3)")
plt.legend()
plt.savefig("lunar_lander_results.png")
print("Plot saved to lunar_lander_results.png")
