"""
CartPole DQN Implementation — Fixed & Stable Version

This project implements a Deep Q-Network (DQN) agent from scratch using PyTorch
to solve the CartPole-v1 environment from OpenAI Gymnasium.

Concepts Covered:
- Deep Q-Network (DQN) Architecture
- Experience Replay Buffer
- Target Network & Periodic Weight Updates
- Epsilon-Greedy Exploration Strategy
- Gradient Clipping for Training Stability
- Neural Network-based Q-Value Approximation

Environment:
- CartPole-v1 (OpenAI Gymnasium)
- State Space : 4 continuous values (cart position, cart velocity,
                pole angle, pole angular velocity)
- Action Space: 2 discrete actions (push left, push right)
- Solved When : 50-episode average reward >= 475

Key Hyperparameters:
- Discount Factor (gamma)    : 0.99
- Initial Epsilon            : 1.0
- Epsilon Decay              : 0.995
- Minimum Epsilon            : 0.01
- Learning Rate              : 0.0005  ← stabilized
- Replay Buffer Capacity     : 50,000  ← larger memory
- Batch Size                 : 64
- Target Network Update      : Every 20 episodes ← more stable
- Gradient Clipping          : 1.0 ← prevents exploding gradients

Author    : Rikesh Yadav
Lab       : INEC Laboratory, Yuan Ze University, Taiwan
Program   : International Internship Pilot Program (IIPP)
Supervisor: Dr. Ihsan Ullah
Date      : June 2026
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# ── 1. Neural Network ────────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.network(x)

# ── 2. Replay Buffer ─────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# ── 3. DQN Agent ─────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma         = 0.99
        self.epsilon       = 1.0
        self.epsilon_min   = 0.05
        self.epsilon_decay = 0.997
        self.lr            = 0.0005
        self.batch_size    = 64
        self.target_update = 20

        self.q_network      = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.memory    = ReplayBuffer(capacity=50000)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch                            = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states))
        actions     = torch.LongTensor(actions)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones       = torch.FloatTensor(dones)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0]
            target_q   = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# ── 4. Training Loop ─────────────────────────────────────────────────
def train():
    env         = gym.make('CartPole-v1')
    state_size  = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent    = DQNAgent(state_size, action_size)
    episodes = 6000
    scores   = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(500):
            action                              = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done                                = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            agent.learn()

            state        = next_state
            total_reward += reward

            if done:
                break

        if episode % agent.target_update == 0:
            agent.update_target()

        scores.append(total_reward)
        avg = np.mean(scores[-50:])

        print(f"Episode {episode+1:3d} | "
              f"Score: {total_reward:6.1f} | "
              f"Avg(50): {avg:6.1f} | "
              f"ε: {agent.epsilon:.3f}")

        if avg >= 475:
            print(f"\n✅ SOLVED at episode {episode+1}!")
            break

    env.close()

    # ── Plot ─────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    plt.plot(scores, alpha=0.4, color='steelblue', label='Score per episode')
    plt.plot(np.convolve(scores, np.ones(50)/50, mode='valid'),
             color='darkorange', linewidth=2.5, label='50-episode average')
    plt.axhline(y=475, color='red', linestyle='--',
                linewidth=1.5, label='Solved threshold (475)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('DQN Training on CartPole-v1 — Fixed & Stable')
    plt.legend()
    plt.tight_layout()
    plt.savefig('dqn_results_fixed.png', dpi=150)
    plt.show()
    print("📊 Plot saved as dqn_results_fixed.png")

if __name__ == "__main__":
    train()