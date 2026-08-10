# 🤖 CartPole DQN — Deep Q-Network from Scratch

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![Internship](https://img.shields.io/badge/INEC%20Lab-Yuan%20Ze%20University-orange.svg)

A clean, beginner-friendly implementation of **Deep Q-Network (DQN)** using **PyTorch** to solve the classic **CartPole-v1** reinforcement learning environment. Built as part of research training at the INEC Laboratory, Yuan Ze University, Taiwan.

---

## 📌 What is This Project?

This project teaches you — step by step — how an AI agent learns to **balance a pole on a moving cart** using only trial and error. No human tells it what to do. It figures it out by itself through **rewards and penalties**.

This is called **Reinforcement Learning (RL)**, and DQN is one of the most important algorithms in RL history — the same family of algorithms that beat human champions in Atari games.

---

## 🧠 Concepts Covered

| Concept | Simple Explanation |
|---|---|
| **Q-Learning** | A method where the agent learns how "good" each action is in each situation |
| **Deep Q-Network (DQN)** | A neural network that replaces the Q-table when states are too many to list |
| **Experience Replay** | The agent stores past experiences and learns from random samples — like reviewing old memories |
| **Target Network** | A frozen copy of the neural network used to stabilize learning |
| **Epsilon-Greedy Policy** | Early on, the agent explores randomly. Over time, it trusts its own knowledge more |
| **Temporal Difference Learning** | The agent updates its knowledge after every single step, not just at the end |

---

## 🎮 The Environment — CartPole-v1

```
        |
       /|\
      / | \
     /  |  \
────[  CART  ]────
→ → → → → → → → →
```

| Property | Details |
|---|---|
| **Goal** | Keep the pole upright as long as possible |
| **State** | 4 continuous values: cart position, cart velocity, pole angle, pole angular velocity |
| **Actions** | 2 discrete: Push LEFT or Push RIGHT |
| **Reward** | +1 for every timestep the pole stays upright |
| **Solved** | Average reward ≥ 475 over last 50 episodes |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      DQN ARCHITECTURE                   │
│                                                         │
│   State (4)  →  Linear(128)  →  ReLU                   │
│              →  Linear(128)  →  ReLU                   │
│              →  Linear(2)    →  Q-values [Left, Right] │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                        │
│                                                         │
│  1. Observe state s                                     │
│  2. Choose action (ε-greedy: explore or exploit)        │
│  3. Execute action → get reward r, next state s'        │
│  4. Store (s, a, r, s', done) in Replay Buffer          │
│  5. Sample random batch from buffer                     │
│  6. Compute target: y = r + γ · max Q_target(s', a')   │
│  7. Update Q-network: minimize (y - Q(s,a))²           │
│  8. Every 10 episodes: copy Q-network → Target Network  │
│  9. Decay ε (explore less over time)                    │
│  10. Repeat until solved ✅                             │
└─────────────────────────────────────────────────────────┘
```

---

## ⚙️ Key Hyperparameters

| Parameter | Value | Why This Value? |
|---|---|---|
| Discount Factor (γ) | 0.99 | Agent cares a lot about future rewards |
| Initial Epsilon (ε) | 1.0 | Start with full random exploration |
| Epsilon Decay | 0.995 | Slowly reduce exploration each episode |
| Minimum Epsilon | 0.01 | Always keep 1% random exploration |
| Learning Rate | 0.001 | Small steps → stable learning |
| Replay Buffer Size | 10,000 | Large enough memory to break correlations |
| Batch Size | 64 | Standard mini-batch for stable gradients |
| Target Update Frequency | Every 10 episodes | Frequent enough to track, slow enough to stabilize |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run Training

```bash
run_week.cmd
```

### Expected Output

```
Episode   1 | Score:   12.0 | Avg(50):  12.0 | ε: 0.995
Episode   2 | Score:   18.0 | Avg(50):  15.0 | ε: 0.990
...
Episode 187 | Score:  475.0 | Avg(50): 476.2 | ε: 0.010
✅ SOLVED at episode 187!
📊 Plot saved as dqn_results_fixed.png
```

---

## 📊 Training Results

After training, a reward plot is automatically saved as `dqn_results.png` showing:

- **Blue (transparent):** Raw score per episode
- **Orange line:** 50-episode rolling average
- **Red dashed line:** Solved threshold (475)

The agent typically solves CartPole-v1 within **150–250 episodes**.

---

## 📁 Project Structure

```
cartpole-dqn/
│
├── CartPole.py           # Main implementation
│   ├── DQN               # Neural network class
│   ├── ReplayBuffer      # Experience replay memory
│   ├── DQNAgent          # Agent with act(), learn(), update_target()
│   └── train()           # Training loop + plotting
│
├── dqn_results_fixed.png # Training reward plot (auto-generated)
└── README.md             # This file
```

---

## 🔬 Why DQN? — The Story

Before DQN, Q-Learning used a **table** to store the value of every (state, action) pair. This works fine for simple problems like GridWorld. But CartPole has **infinite possible states** (continuous position, velocity values) — you can't build a table for that.

**DQN's solution:** Replace the table with a **neural network** that approximates Q-values for any state it has never seen before. This was first proposed by DeepMind in 2013 and is the foundation of modern Deep Reinforcement Learning.

Two key innovations that make DQN stable:

1. **Experience Replay** — Breaks the correlation between consecutive training samples, making learning more stable and data-efficient.
2. **Target Network** — Prevents the "chasing a moving target" problem by keeping a frozen copy of the network for computing TD targets.

---

## 🔭 What's Next?

This project is **Week 2** of a structured 12-week research plan. The roadmap ahead:

```
Week 1  ✅  Q-Learning & GridWorld
Week 2  ✅  DQN on CartPole-v1          ← You are here
Week 3  🔜  Experience Replay + Target Network (advanced)
Week 4  🔜  LunarLander-v3 + Hyperparameter Tuning
Week 5–8    Graph Convolutional Networks (GCN)
Week 9–12   GCN-Enhanced DQN for IoT Edge Computing Research
```

**Research Goal:** Develop a novel **GCN-Enhanced DQN** framework for joint latency and energy-aware task offloading in heterogeneous IoT edge networks — targeting IEEE conference publication.

---

## 👨‍💻 Author

**Rikesh Yadav**
Computer Science & Engineering (7th Semester)
Sri Eshwar College of Engineering, India

**Research Intern** — International Internship Pilot Program (IIPP)
**Laboratory:** Intelligent Networks and Edge-Cloud Computing (INEC Lab)
**Institution:** Yuan Ze University, Taoyuan, Taiwan
**Supervisor:** Assistant Professor Dr. Ihsan Ullah
**Internship Period:** June – August 2026

📧 GitHub: [github.com/Rieck-Ace](https://github.com/Rieck-Ace)

---

## 📚 References

- Mnih, V. et al. (2013). *Playing Atari with Deep Reinforcement Learning.* DeepMind.
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature, 518, 529–533.
- OpenAI Gymnasium: [gymnasium.farama.org](https://gymnasium.farama.org)
- PyTorch Documentation: [pytorch.org](https://pytorch.org)

---

> *"An agent that learns from experience is more powerful than one that is programmed with rules."*
> — Reinforcement Learning Philosophy

---

⭐ If this helped you understand DQN, consider starring the repository!