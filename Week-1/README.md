# Week 1 - GridWorld Q-Learning

This project implements a simple **GridWorld environment** and a **Q-Learning agent** from scratch to understand the core concepts of **Reinforcement Learning (RL)**.

The implementation was developed as part of my learning journey during the **International Internship Pilot Program (IIPP)** at the **Intelligent Networks & Edge-Cloud Computing (INEC) Laboratory, Yuan Ze University, Taiwan**.

---

## Project Overview

The objective is to train an agent to navigate a 3×3 grid and reach the goal state while avoiding obstacles and invalid moves.

The agent learns through interaction with the environment using the **Q-Learning algorithm**, gradually improving its policy based on rewards received from previous experiences.

---

## Environment Setup

### Grid Layout

```text
S . .
. X .
. . G
```

Where:

* **S** = Start State
* **G** = Goal State
* **X** = Obstacle
* **A** = Agent

---

## State Space

The environment contains 9 states represented as grid coordinates:

```text
(0,0) (0,1) (0,2)

(1,0) (1,1) (1,2)

(2,0) (2,1) (2,2)
```

---

## Action Space

The agent can perform four actions:

```text
UP
DOWN
LEFT
RIGHT
```

---

## Reward Structure

The reward function is designed to encourage the agent to reach the goal while avoiding inefficient behavior.

| Event                       | Reward |
| --------------------------- | ------ |
| Normal Move                 | -1     |
| Invalid Move (Boundary Hit) | -5     |
| Obstacle Collision          | -10    |
| Goal Reached                | +10    |

---

## Q-Learning Components

The implementation includes:

* Q-Table initialization
* Epsilon-Greedy action selection
* Exploration vs Exploitation strategy
* Q-Value updates
* Episode-based training
* Policy evaluation after training

### Hyperparameters

```python
alpha = 0.1      # Learning Rate
gamma = 0.9      # Discount Factor
epsilon = 1.0    # Initial Exploration Rate
```

---

## Learning Process

The agent follows the Reinforcement Learning cycle:

```text
Current State
      ↓
Choose Action
      ↓
Interact with Environment
      ↓
Receive Reward
      ↓
Observe Next State
      ↓
Update Q-Value
      ↓
Repeat
```

Over multiple episodes, the agent learns which actions maximize cumulative rewards and eventually discovers an optimal path to the goal.

---

## Features Implemented

* Custom GridWorld Environment
* State and Action Space Design
* Reward Engineering
* Obstacle Handling
* Goal State Detection
* Epsilon-Greedy Exploration
* Q-Learning Training Loop
* Learned Policy Execution
* Grid Visualization

---

## How to Run

```bash
python gridWorld.py
```

---

## Concepts Covered

This project demonstrates the practical implementation of:

* Reinforcement Learning
* Q-Learning
* Markov Decision Processes (MDP)
* Exploration vs Exploitation
* Learning Rate (Alpha)
* Discount Factor (Gamma)
* Reward Functions
* Policy Learning
* State Transitions

---

## Future Improvements

Potential enhancements include:

* Larger GridWorld environments
* Multiple obstacles and goals
* Dynamic environments
* Reward tracking and visualization
* Q-Table persistence
* Performance analytics
* Deep Q-Networks (DQN)
* PyTorch implementation
* Experience Replay
* Target Networks

---

## Author

**Rikesh Yadav**

Developed as part of the Reinforcement Learning learning track during the **IIPP Research Internship at INEC Laboratory, Yuan Ze University, Taiwan (2026)**.
