# Weekly Meeting Notes

## Meeting 1
**Agenda**: Introduction, Progress Review, and Research Direction
**Discussion Points**
- Presented my previous projects and research-related work completed before the internship.
- Discussed the strengths and limitations of the existing work to identify opportunities for developing a more valuable research contribution.
- Received guidance on strengthening the fundamental concepts of Reinforcement Learning and Machine Learning before moving to advanced implementations.
**Outcome**
- Established a clear learning roadmap for the internship.
- Decided to focus on building strong theoretical foundations before proceeding to implementation and research development.

## Meeting 2
**Agenda**: Progress Review and Research Planning
**Discussion Points**
- Presented the work completed during Week 1 and Week 2, including Reinforcement Learning fundamentals, Q-Learning, GridWorld implementation, and initial Deep Q-Network (DQN) concepts.
- Demonstrated the implementation results and discussed the learning outcomes.
- Received suggestions to incorporate advanced Reinforcement Learning techniques such as Deep Q-Networks (DQN) and Deep Reinforcement Learning (DRL) into the proposed research work.
- Collaborated on the research manuscript using Overleaf, discussing improvements to the paper structure and technical content.
**Outcome**
- Continued implementation of advanced RL concepts.
- Refined the research direction to align with modern DRL-based approaches.

## Meeting 3 (Online Meeting – Due to Typhoon)
**Agenda**: CartPole-v1 Implementation Review
**Discussion Points**
- Reviewed the implementation and training results of the CartPole-v1 environment.
- Analyzed the learning performance of the agent and identified issues affecting convergence and policy learning.
- Received guidance on improving hyperparameter tuning and training stability to enable the agent to learn more effectively.
**Outcome**
- Planned further experiments by adjusting key training parameters and improving the DQN implementation.

## Meeting 4
**Date**: 17 July 2026 (Friday)
**Agenda**: Mid-Term Report Review and Research Implementation Planning
**Discussion Points**
- Reviewed the submitted Mid-Term Internship Report and provided feedback on the overall structure, technical content, and presentation.
- Suggested improvements to the research paper, including revisions to methodology, implementation details, and documentation.
- Guided the initial implementation phase of the proposed research by outlining the development workflow, including:
  - Environment setup
  - Sensor integration
  - Preparation of dummy or real datasets (where available)
  - Implementation and execution of the proposed algorithms
**Outcome**
- Identified the necessary modifications for both the report and research paper.
- Established the implementation roadmap for the next phase of the internship (Adaptive Evacuation Environment).

## Meeting 5
**Agenda**: Phase 1 Environment Design & Grid Mechanics
**Discussion Points**
- Presented the initial concept for the Adaptive Evacuation Environment.
- The Professor recommended adopting the standard Gymnasium API to ensure the environment is compatible with modern RL libraries.
- Discussed the mechanics of dynamic hazards. The Professor suggested using Cellular Automata for fire and smoke spreading to simulate realistic, unpredictable environments rather than static obstacles.
- Reviewed the multi-objective reward structure (balancing speed vs. safety).
**Outcome**
- Successfully implemented the 10x10 procedural grid with dynamic fire spread.
- Prepared the environment for the baseline DQN agent integration.

## Meeting 6
**Agenda**: Baseline DQN Review & Graph Neural Network Transition
**Discussion Points**
- Demonstrated the Baseline Double DQN agent successfully navigating the 10x10 environment.
- The Professor pointed out a fundamental limitation in the approach: the Flat-MLP neural network memorizes the 10x10 grid (800-dimension input) and will instantly crash if deployed in a larger building.
- The Professor recommended shifting the research focus toward Graph Neural Networks (GNNs) using PyTorch Geometric. This would allow the agent to learn spatial relationships (message passing) and generalize to any grid size natively.
**Outcome**
- Implemented Phase 3 (GNN-DQN).
- Verified zero-shot transfer capabilities (evaluating the 10x10 trained agent on a 15x15 grid successfully).

## Meeting 7
**Agenda**: GNN Results Analysis & Hybrid Heuristic Integration
**Discussion Points**
- Reviewed the comparative experiment results between the Baseline DQN and the GNN-DQN.
- While the GNN achieved 100% transferability, the Professor noted its slow training convergence (938 episodes) due to the complexity of learning message-passing from scratch.
- The Professor strongly recommended exploring a "Hybrid" approach: combining classical planning algorithms (like A* or Dijkstra) with the Deep RL agent.
- Discussed how A* could calculate the shortest path to the exit and provide it to the GNN as a "hint", leaving the RL agent responsible only for dodging the spreading fire.
**Outcome**
- Implemented Phase 4 (Hybrid GNN-A*).
- Achieved a 26% faster convergence rate (691 episodes) compared to the pure GNN, officially validating the hybrid research hypothesis.

---

## Overall Progress Summary

| Meeting | Focus Area | Key Outcome |
| :--- | :--- | :--- |
| **Meeting 1** | Introduction & Research Planning | Established learning roadmap and research direction. |
| **Meeting 2** | RL Progress Review | Introduced DQN/DRL concepts and improved research planning. |
| **Meeting 3** | CartPole-v1 Review | Identified improvements for DQN training and hyperparameter tuning. |
| **Meeting 4** | Mid-Term Review & Implementation | Finalized implementation roadmap and report improvement plan. |
| **Meeting 5** | Environment Design (Phase 1) | Built Gymnasium-compatible Cellular Automata fire grid. |
| **Meeting 6** | GNN Transition (Phases 2 & 3) | Replaced DQN with PyG GNN for zero-shot spatial transferability. |
| **Meeting 7** | Hybrid Integration (Phase 4) | Integrated A* heuristic with GNN, improving training efficiency by 26%. |
