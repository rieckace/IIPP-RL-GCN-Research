# 🚀 Week 4 - LunarLander-v3 (Scaling Up the DQN)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)
![Internship](https://img.shields.io/badge/INEC%20Lab-Yuan%20Ze%20University-orange.svg)

This week, we take our **Improved DQN (with Experience Replay & Target Networks)** from Week 3 and test it on a significantly harder environment: **LunarLander-v3**. 

Balancing a pole is relatively easy, but safely landing a spacecraft on a targeted landing pad requires complex sequence planning and precise thruster control!

---

## 🛸 The Environment: LunarLander-v3

In this environment, the agent controls a lander module attempting to land on a pad at coordinate (0,0). 

| Property | Details |
|---|---|
| **State Space (8)** | X/Y coordinates, X/Y velocities, Angle, Angular velocity, Left/Right leg ground contact (Booleans). |
| **Action Space (4)** | 0: Do nothing, 1: Fire left engine, 2: Fire main engine, 3: Fire right engine. |
| **Reward** | +100 to +140 for moving to landing pad. -100 for crashing. +10 for each leg grounding. Firing main engine costs -0.3 points. |
| **Solved Status** | Average score of 200 over 100 consecutive episodes. |

Because the state space is twice as large as CartPole and the action space is more complex, a naive DQN struggles. We need to **tune our hyperparameters**.

---

## ⚙️ Hyperparameter Tuning

To make the DQN learn this complex task, I adjusted several key settings from last week:

1. **Larger Neural Network:** 
   - CartPole: `(4 -> 128 -> 128 -> 2)`
   - LunarLander: `(8 -> 256 -> 256 -> 4)`
   - *Why?* More parameters are needed to understand the complex flight dynamics.

2. **Learning Rate (LR):** 
   - Changed from `1e-3` to `5e-4`. 
   - *Why?* LunarLander rewards are highly variable (crashing vs. landing). A smaller LR prevents the network from taking huge, unstable gradient steps when it crashes.

3. **Epsilon Decay:**
   - Changed from `0.995` to `0.99`.
   - *Why?* We want the agent to explore a lot initially to find the landing pad, but then rapidly stop exploring so it doesn't accidentally fire thrusters and crash during exploitation.

4. **Replay Buffer Size:**
   - Increased to `100,000`.
   - *Why?* LunarLander episodes are longer, and learning the rare "successful landing" requires storing many past experiences.

---

## 💻 Implementation Highlights

```python
# Expanded Neural Network for LunarLander
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
```

We also introduced `env = gym.make('LunarLander-v3')`. Note: You may need to install Box2D physics engine: `pip install gymnasium[box2d]`.

---

## 📊 Results & Observations

- **Early Episodes (0-100):** The lander crashes constantly or flies off the screen. Rewards are heavily negative (around -200).
- **Middle Episodes (100-300):** The agent learns to hover to avoid crashing, but struggles to land on the exact pad.
- **Late Episodes (400+):** The agent learns that gently touching down on the pad yields massive positive rewards (+200).

### How to Run
```bash
run_lunar_lander.cmd
```
After training, it will generate a `lunar_lander_results.png` graph plotting the learning curve.

---

## 🔭 What's Next?
We have successfully built a robust DRL agent capable of solving classic control environments. Next week, we transition into **Graph Neural Networks (GCNs)**—the crucial missing piece for our adaptive evacuation research!

```
Week 1  ✅  Q-Learning & GridWorld
Week 2  ✅  Basic DQN on CartPole-v1
Week 3  ✅  Experience Replay + Target Network
Week 4  ✅  LunarLander-v3 + Hyperparameter Tuning    ← You are here
Week 5  🔜  Learn About Graphs & PyTorch Geometric
Week 6-8    Build and Use First GCN
```

---
## 👨‍💻 Author
**Rikesh Yadav**  
Research Intern — International Internship Pilot Program (IIPP)  
**Laboratory:** Intelligent Networks and Edge-Cloud Computing (INEC Lab)  
**Institution:** Yuan Ze University, Taoyuan, Taiwan  
