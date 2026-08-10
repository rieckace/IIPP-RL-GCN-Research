# 🚀 Week 3 - Improving the DQN (Experience Replay & Target Network)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)
![Internship](https://img.shields.io/badge/INEC%20Lab-Yuan%20Ze%20University-orange.svg)

This week, we take the basic Deep Q-Network (DQN) built in Week 2 and upgrade it with two critical enhancements: **Experience Replay** and a **Target Network**. These are the exact features that allowed DeepMind to achieve superhuman performance on Atari games!

---

## 📖 Theory & Core Concepts

In Week 2, our agent learned directly from its immediate experiences. However, that approach has two major flaws:
1. **Correlated Data:** Sequential states in a game are highly related. If the agent only learns from consecutive frames, the neural network forgets past lessons and overfits to the current situation.
2. **Moving Targets:** The network uses its own Q-values to calculate the target for its updates. This is like trying to hit a target that moves every time you take a shot!

### 1. Experience Replay 🧠
Instead of learning from an experience and immediately throwing it away, the agent stores all experiences `(state, action, reward, next_state, done)` in a large memory buffer (e.g., holding 10,000 transitions). 
During training, the agent samples a **random mini-batch** from this buffer.
- **Why it works:** It breaks the correlation between consecutive steps and allows the agent to learn from rare but important past experiences multiple times.

### 2. Target Network 🎯
We create a second neural network called the **Target Network**. It is an exact clone of the main Q-Network, but its weights are **frozen**. We use this frozen network to calculate the target Q-value: `Target = Reward + Gamma * max(Target_Q(next_state))`
- Every 10 or 20 episodes, we copy the weights from the main network to the target network.
- **Why it works:** It provides a stable target for the loss function, preventing the training from oscillating wildly.

---

## 💻 Implementation Details

In `improved_dqn.py`, we implement these features on top of the CartPole environment:

```python
# The Replay Buffer (Memory)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        # Random sampling breaks correlation!
        transitions = random.sample(self.buffer, batch_size)
        return zip(*transitions)
```

```python
# The Target Update Logic
if episode % TARGET_UPDATE == 0:
    target_net.load_state_dict(policy_net.state_dict())
```

---

## 📊 Testing and Comparison

When comparing the Week 2 basic DQN with the Week 3 Improved DQN:
- **Stability:** The Improved DQN does not suffer from "catastrophic forgetting" where the score suddenly crashes to 0 after doing well.
- **Speed:** The agent solves the environment faster and with much less variance between episodes.

### How to Run
```bash
run_week.cmd
```

---

## 🔭 What's Next?
```
Week 1  ✅  Q-Learning & GridWorld
Week 2  ✅  Basic DQN on CartPole-v1
Week 3  ✅  Experience Replay + Target Network    ← You are here
Week 4  🔜  LunarLander-v3 + Hyperparameter Tuning
Week 5–8    Graph Convolutional Networks (GCN)
```

---
## 👨‍💻 Author
**Rikesh Yadav**  
Research Intern — International Internship Pilot Program (IIPP)  
**Laboratory:** Intelligent Networks and Edge-Cloud Computing (INEC Lab)  
**Institution:** Yuan Ze University, Taoyuan, Taiwan  
