# 🏆 Week 12 - Final Project (Adaptive Evacuation GNN-DQN)

![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)
![Internship](https://img.shields.io/badge/INEC%20Lab-Yuan%20Ze%20University-orange.svg)

This is the final week of the 3-Month IIPP Study Plan! 
Over the past 11 weeks, we have evolved from basic Q-Learning GridWorlds to sophisticated hybrid Graph Neural Networks.

It is now time to apply everything we've learned to a real-world, high-impact research problem.

---

## 🎯 The Final Project: Intelligent Building Evacuation

While the original syllabus suggested task offloading or UAV pathfinding, I chose to focus my final project on **IoT-enabled Smart Building Evacuation during Fire Emergencies**. 

This directly leverages the exact `GCN-DQN` architecture we developed in Week 10!

### Why this problem?
1. **Dynamic Environments:** Fires spread unpredictably. A standard pathfinding algorithm (A*) will lead evacuees into deadly traps if the fire changes between calculations.
2. **Generalization:** If we train a model on an Office layout, it must be able to deploy in a Hospital without retraining. Only our Graph-based approach allows this.

---

## 🏗️ Project Architecture Overview

The complete implementation for this final project is massive and has been built out in its own dedicated directory in this repository:
**➡️ [`Adaptive-Evacuation-GNN-DQN/`](../Adaptive-Evacuation-GNN-DQN/)**

### What we implemented in the main project:
1. **The Grid-to-Graph Engine (Week 8 concepts):** Dynamically mapping live IoT occupancy and thermal sensors into PyTorch Geometric graphs.
2. **Hazard-Aware Reward Shaping (Week 2/3 concepts):** Using dynamic A* potential fields as dense rewards for the DQN, masking active fires as impassable walls.
3. **The GCN-DQN (Week 9-11 concepts):** Utilizing Agent-Centric Pooling to isolate the agent's exact node embedding from the GCN, feeding it to the DQN action head.

## ▶️ How to Run
```bash
run_week.cmd
```
This wrapper points to the main `Adaptive-Evacuation-GNN-DQN/` project and prints the available train, evaluate, and dashboard commands.

### Requirements
Install the final-project stack from `../Adaptive-Evacuation-GNN-DQN/requirements.txt`, plus the dashboard packages used by the wrapper:
```bash
pip install fastapi uvicorn pytest
```

---

## 📈 Final Results Summary
The GNN-DQN agent was trained on a 10x10 Office layout for 1500 episodes.
When dropped into a highly complex, unseen **22x22 Hospital Layout** and an enormous **30x30 Shopping Mall**, the agent successfully generalized its spatial knowledge and achieved a **100% successful evacuation rate** zero-shot!

The project also includes a complete React/FastAPI real-time dashboard for IoT visualization.

---

## 🎓 Internship Conclusion

This concludes the 12-week study plan. The progression from basic Q-Tables to state-of-the-art Hybrid Graph Reinforcement Learning has been an incredible journey. The findings from this final project are currently being drafted into a formal academic paper for the **IEEE International Conference on Intelligent Environments (IEEE IE 2027)**.

Thank you to Dr. Ihsan Ullah and the INEC Lab for the guidance and resources during this internship!

```
Week 1-4  ✅  RL Basics & Deep Q-Networks
Week 5-8  ✅  PyTorch Geometric & Custom Graphs
Week 9-11 ✅  Hybrid GCN-DQN Architecture
Week 12   ✅  Final Project: Adaptive Evacuation GNN-DQN  🎉
```

---
## 👨‍💻 Author
**Rikesh Yadav**  
Research Intern — International Internship Pilot Program (IIPP)  
**Laboratory:** Intelligent Networks and Edge-Cloud Computing (INEC Lab)  
**Institution:** Yuan Ze University, Taoyuan, Taiwan  
