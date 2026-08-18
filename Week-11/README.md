# 🚂 Week 11 - Training and Testing the Hybrid Agent

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-2.3+-green.svg)
![Internship](https://img.shields.io/badge/INEC%20Lab-Yuan%20Ze%20University-orange.svg)

With the architecture written, this week we focus on **Training and Testing**. How does a GCN-DQN compare to a standard DQN on a spatial navigation task?

---

## 📊 The Comparison: Standard DQN vs GCN-DQN

If we train both agents to navigate a 5x5 GridWorld with obstacles, they will both learn to reach the goal. 

However, the real test is **Zero-Shot Transfer**. What happens if we drop the trained agents into a larger 10x10 grid with a different wall layout?

### Standard DQN (The Failure)
- The Standard DQN expects exactly 25 inputs (a 5x5 flattened grid).
- When given a 10x10 grid (100 inputs), the Neural Network throws a shape mismatch error and crashes instantly.
- Even if we pad the grid with zeros to force it to fit, the agent's spatial understanding is shattered. It just blindly hits walls.

### GCN-DQN (The Success)
- The GCN-DQN does not care about the size of the grid. It only processes the graph (Nodes and Edges).
- When dropped into the 10x10 grid, it successfully maps the new walls and corridors into a new graph.
- Because the GCN learned the *local topology* (e.g., "moving away from walls is good"), it successfully navigates the new maze without any additional training!

---

## 💻 Implementation: Training Script

The training loop for a GCN-DQN is identical to the Improved DQN we wrote in Week 3, with one exception: **The Replay Buffer stores Graphs, not arrays!**

Instead of a NumPy array of pixels, we store PyG `Data` objects.

```python
# During the step loop:
current_graph = env.get_graph_state()
action = select_action(current_graph)
next_graph, reward, done = env.step(action)

# Push to memory
memory.push(current_graph, action, reward, next_graph, done)
```

During the learning phase, we use PyTorch Geometric's `DataLoader` to batch the graphs together into giant disconnected super-graphs for fast GPU processing.

---

## ▶️ How to Run
```bash
run_week.cmd
```
This runs the Week 11 graph replay-buffer simulation.

## 🔭 What's Next?
We have proven that GCN-DQN is superior for spatial generalization. 
Next week is the final week of the internship! We will define a real-world problem and use this architecture to solve it in our **Final Mini Project**.

```
Week 1-9  ✅  RL Basics, GCNs, and Hybrid Theory
Week 10   ✅  Build GCN-DQN Model
Week 11   ✅  Train and Test                         ← You are here
Week 12   🔜  Final Mini Project (Adaptive Evacuation!)
```

---
## 👨‍💻 Author
**Rikesh Yadav**  
Research Intern — International Internship Pilot Program (IIPP)  
**Laboratory:** Intelligent Networks and Edge-Cloud Computing (INEC Lab)  
**Institution:** Yuan Ze University, Taoyuan, Taiwan  
