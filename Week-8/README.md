# 🗺️ Week 8 - Converting Spatial Grids to Dynamic Graphs

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-2.3+-green.svg)
![Internship](https://img.shields.io/badge/INEC%20Lab-Yuan%20Ze%20University-orange.svg)

This week concludes our deep dive into Graph Neural Networks (GCNs) by focusing on **dynamic graph construction**. In Week 7, we built a static IoT network. This week, we learn how to convert a 2D spatial grid (like a building floorplan) into a graph format.

This is the exact technique we will use for our final internship project: **Adaptive Evacuation**!

---

## 📖 Theory: Why convert a Grid to a Graph?

If an agent is navigating a building, we could just feed the 2D map into a CNN (Convolutional Neural Network). However, CNNs struggle when the dimensions of the grid change (e.g., training on a 10x10 room, then deploying in a 30x30 mall). 

If we convert the grid into a Graph:
- **Nodes = Floor Tiles (1x1 meter cells)**
- **Edges = Valid walking paths (Up, Down, Left, Right)**
- **Features = (Is_Wall, Is_Fire, Is_Agent, Is_Exit)**

A GCN can process this graph no matter how big the building is! It only cares about local node connections, making it **scale-invariant**.

---

## 💻 Implementation: Grid-to-Graph

In `dynamic_graph.py`, we write a script that takes a simple 2D text maze and automatically generates the PyTorch Geometric `Data` object.

### The Algorithm:
1. Loop through every `(row, col)` in the 2D grid.
2. If the cell is not a wall, create a Node.
3. Check orthogonal neighbors (Up, Down, Left, Right).
4. If a neighbor is also not a wall, create an Edge connecting them in the `edge_index`.

```python
# Pseudo-code for edge generation
if grid[r][c] != WALL:
    if r > 0 and grid[r-1][c] != WALL:
        edges.append([node_id, top_node_id])
    if c > 0 and grid[r][c-1] != WALL:
        edges.append([node_id, left_node_id])
```

---

## 🔭 What's Next?
We now have the skills to build Deep Q-Networks (Week 1-4) and we know how to model spatial environments as Graphs (Week 5-8). 

Starting next week, we merge these two fields to create the **GCN-DQN**, the architecture that will power our final Evacuation AI!

## ▶️ How to Run
```bash
run_week.cmd
```
This prints the grid-to-graph conversion demo in the terminal.

```
Week 5-8  ✅  Learn About Graphs & Grid-to-Graph Conversion
Week 9    🔜  Learn the Idea: Combining DQN and GCN
Week 10   🔜  Build GCN-DQN Model
Week 11   🔜  Train and Test
Week 12   🔜  Final Mini Project (Adaptive Evacuation!)
```

---
## 👨‍💻 Author
**Rikesh Yadav**  
Research Intern — International Internship Pilot Program (IIPP)  
**Laboratory:** Intelligent Networks and Edge-Cloud Computing (INEC Lab)  
**Institution:** Yuan Ze University, Taoyuan, Taiwan  
