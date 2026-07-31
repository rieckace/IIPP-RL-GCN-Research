import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from environment.make_env import make_env
from models.dqn.trainer import DQNAgent
from experiments.run_experiments import run_evaluation

def check_dqn():
    env = make_env("apartment")
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    agent = DQNAgent(config)
    agent.load_checkpoint("checkpoints/dqn/best_model.pt")
    agent.epsilon = 0.0 # Force greedy for evaluation
    
    stats = run_evaluation(agent, env, num_episodes=10, max_steps=200)
    print("DQN Evaluation Results on Apartment:")
    print(stats)

if __name__ == "__main__":
    check_dqn()
