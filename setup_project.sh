Adaptive-Evacuation-GNN-DQN/
│
├── README.md                          # Project overview
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
├── main.py                            # Main project entry point
│
├── .vscode/
│   ├── settings.json
│   ├── launch.json
│   └── tasks.json
│
├── configs/
│   ├── config.py
│   ├── environment.yaml
│   ├── dqn.yaml
│   ├── gnn.yaml
│   ├── training.yaml
│   └── experiment.yaml
│
├── knowledge/
│   ├── daily_notes.md
│   ├── algorithms.md
│   ├── ideas.md
│   ├── bugs.md
│   ├── research_questions.md
│   └── future_work.md
│
├── docs/
│   ├── project_plan.md
│   ├── methodology.md
│   ├── weekly_progress.md
│   ├── meeting_notes.md
│   ├── experiment_log.md
│   └── paper_notes.md
│
├── assets/
│   ├── diagrams/
│   ├── images/
│   ├── icons/
│   └── videos/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── synthetic/
│   └── graphs/
│
├── checkpoints/
│   ├── dqn/
│   ├── gnn/
│   └── hybrid/
│
├── environment/
│   ├── __init__.py
│   │
│   ├── maps/
│   │   ├── map_v1.py
│   │   ├── map_v2.py
│   │   └── map_v3.py
│   │
│   ├── entities/
│   │   ├── agent.py
│   │   ├── exit.py
│   │   ├── obstacle.py
│   │   ├── hazards.py
│   │   ├── smoke.py
│   │   └── sensors.py
│   │
│   ├── evacuation_env.py
│   ├── building.py
│   ├── grid.py
│   ├── actions.py
│   ├── reward.py
│   ├── state.py
│   ├── renderer.py
│   └── utils.py
│
├── models/
│   │
│   ├── common/
│   │   ├── layers.py
│   │   ├── utils.py
│   │   └── losses.py
│   │
│   ├── dqn/
│   │   ├── network.py
│   │   ├── replay_buffer.py
│   │   ├── target_network.py
│   │   ├── trainer.py
│   │   └── inference.py
│   │
│   ├── gnn/
│   │   ├── gcn.py
│   │   ├── graph_builder.py
│   │   ├── graph_dataset.py
│   │   └── graph_utils.py
│   │
│   └── hybrid/
│       ├── gnn_dqn.py
│       ├── trainer.py
│       └── inference.py
│
├── training/
│   ├── train_dqn.py
│   ├── train_gnn.py
│   ├── train_hybrid.py
│   ├── evaluate.py
│   └── hyperparameter_search.py
│
├── evaluation/
│   ├── metrics.py
│   ├── evaluator.py
│   ├── comparison.py
│   └── statistics.py
│
├── visualization/
│   ├── render_environment.py
│   ├── training_plots.py
│   ├── graph_visualizer.py
│   ├── heatmap.py
│   ├── animation.py
│   └── dashboard.py
│
├── experiments/
│   ├── baseline_dqn/
│   ├── dynamic_smoke/
│   ├── sensor_fusion/
│   ├── gnn_integration/
│   └── final_experiments/
│
├── notebooks/
│   ├── Environment_Test.ipynb
│   ├── DQN_Experiments.ipynb
│   ├── GNN_Experiments.ipynb
│   └── Result_Analysis.ipynb
│
├── paper/
│   ├── manuscript/
│   │   ├── manuscript.tex
│   │   ├── references.bib
│   │   └── sections/
│   │       ├── abstract.tex
│   │       ├── introduction.tex
│   │       ├── related_work.tex
│   │       ├── methodology.tex
│   │       ├── experiments.tex
│   │       ├── results.tex
│   │       └── conclusion.tex
│   │
│   ├── figures/
│   ├── tables/
│   └── supplementary/
│
├── results/
│   ├── figures/
│   ├── plots/
│   ├── screenshots/
│   ├── videos/
│   ├── saved_models/
│   └── logs/
│
├── tests/
│   ├── test_environment.py
│   ├── test_agent.py
│   ├── test_rewards.py
│   ├── test_dqn.py
│   └── test_gnn.py
│
├── scripts/
│   ├── create_environment.py
│   ├── generate_graph.py
│   ├── run_training.py
│   ├── evaluate_model.py
│   └── export_results.py
│
└── utils/
    ├── logger.py
    ├── seed.py
    ├── helpers.py
    └── file_utils.py