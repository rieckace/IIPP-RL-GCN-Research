# NSTC IIPP Taiwan Internship - Deep Reinforcement Learning Research

Welcome to the repository for my research internship under the National Science and Technology Council (NSTC) International Internship Pilot Program (IIPP) in Taiwan. 

This repository chronicles a 12-week research journey focused on **Deep Reinforcement Learning (DRL)**, **Graph Neural Networks (GNN)**, and their application to **Adaptive Building Evacuation Systems**.

## 📖 Internship Overview

Over the course of 12 weeks, this internship progressed from fundamental Reinforcement Learning concepts to advanced, state-of-the-art hybrid AI architectures. The research culminated in the development of a novel evacuation system that dynamically routes agents away from spreading hazards (fire and smoke) in real-time.

### Weekly Progression

*   **Week 1 & 2**: Fundamentals of RL, Q-Learning, and environment design. (Classic control tasks like CartPole).
*   **Week 3 & 4**: Deep Q-Networks (DQN) theory, experience replay, target networks, and PyTorch implementations.
*   **Week 5**: Introduction to Graph Neural Networks (GNN) for spatial representations and pathfinding.
*   **Week 6**: Message Passing Neural Networks (MPNN) and GCNs applied to structured grid environments.
*   **Week 7**: Advanced GNN architectures (GAT, GraphSAGE) and dynamic graph processing.
*   **Week 8 & 9**: Hybridizing GCNs with DQNs. Encoding dynamic hazards into graphs for the RL agent.
*   **Week 10**: Multi-Agent Reinforcement Learning (MARL) concepts. Cooperative vs. Competitive environments.
*   **Week 11**: Evaluation metrics, benchmark testing, and environment design for complex evacuation simulations.
*   **Week 12**: Final project consolidation, model tuning, and research report writing.

## 🚀 Final Project: Adaptive Evacuation GNN-DQN

The capstone project of this internship is located in the `Adaptive-Evacuation-GNN-DQN/` directory.

### Project Description
Traditional evacuation systems rely on static exit signs, which can be fatal if the prescribed path is blocked by dynamic hazards like fire or smoke. This project introduces a hybrid **GCN-DQN (Graph Convolutional Network + Deep Q-Network)** architecture. 

The environment is modeled as a dynamic graph where:
*   **Nodes** represent physical grid cells (empty, wall, fire, smoke, exit).
*   **Edges** represent traversable paths.
*   **Node Features** dynamically update as fire spreads.

The GCN processes the spatial dependencies and hazard propagation, passing an enriched spatial embedding to the DQN, which determines the optimal action (Up, Down, Left, Right, Stay) for the evacuating agent.

### Key Features
*   **Dynamic Fire & Smoke Spread**: Hazards probabilistically spread across the grid during the episode.
*   **Real-Time Rerouting**: The agent learns to adapt its path when preferred routes are blocked.
*   **Graph-Based State Representation**: Highly scalable to different floor plans without retraining from scratch.

## 🛠️ Repository Structure

*   `Week-1/` to `Week-12/`: Weekly learning materials, notes, scripts, and checkpoints.
*   `Adaptive-Evacuation-GNN-DQN/`: The complete codebase for the final evacuation project.
*   `Final_Report_Rikesh_Yadav.docx`: The comprehensive final internship report.
*   `IIPP_Term_Report_Rikesh_Yadav.pdf`: The mid-term evaluation report.

## 📜 Acknowledgements

I would like to express my sincere gratitude to the NSTC IIPP program and my advisors in Taiwan for providing this incredible research opportunity, guidance, and resources throughout this internship.
