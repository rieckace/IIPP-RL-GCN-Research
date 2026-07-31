import os
import json
import numpy as np
from tqdm import tqdm
import yaml

from environment.make_env import make_env
from baselines.agents import RandomAgent, BFSAgent, DijkstraAgent, AStarAgent
from models.dqn.trainer import DQNAgent

def run_evaluation(agent, env, num_episodes=50, max_steps=200):
    """Run an agent in the environment and return statistics."""
    success_count = 0
    total_steps = []
    total_rewards = []
    wall_collisions = []
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        episode_reward = 0
        collisions = 0
        
        while not done and steps < max_steps:
            if hasattr(agent, 'act') and hasattr(agent, 'learning_rate'): # RL Agent
                action = agent.act(obs, explore=False)
            else: # Classical Agent
                action = agent.act(obs, env=env)
                
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Simple heuristic for collision: bumped into wall
            if reward == env.reward_config.wall_bump:
                collisions += 1
                
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
        if info.get("reason") == "reached_exit":
            success_count += 1
            
        total_steps.append(steps)
        total_rewards.append(episode_reward)
        wall_collisions.append(collisions)
        
    return {
        "success_rate": success_count / num_episodes,
        "mean_steps": float(np.mean(total_steps)),
        "std_steps": float(np.std(total_steps)),
        "mean_reward": float(np.mean(total_rewards)),
        "mean_collisions": float(np.mean(wall_collisions))
    }

def evaluate_baselines(map_name="office", num_episodes=10):
    """Evaluate classical baselines on a specific map."""
    print(f"\n--- Evaluating Baselines on {map_name} ---")
    env = make_env(map_name)
    
    agents = {
        "Random": RandomAgent(),
        "BFS": BFSAgent(),
        "Dijkstra": DijkstraAgent(),
        "A*": AStarAgent()
    }
    
    results = {}
    for name, agent in agents.items():
        print(f"Running {name}...")
        stats = run_evaluation(agent, env, num_episodes=num_episodes, max_steps=env.max_steps)
        results[name] = stats
        print(f"{name} -> Success Rate: {stats['success_rate']*100:.1f}%, Mean Steps: {stats['mean_steps']:.1f}")
        
    return results

def evaluate_dqn(map_name="office", checkpoint_path="checkpoints/dqn_best.pt", num_episodes=50):
    """Evaluate a trained DQN agent."""
    print(f"\n--- Evaluating DQN on {map_name} ---")
    env = make_env(map_name)
    
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    agent = DQNAgent(config)
    if os.path.exists(checkpoint_path):
        agent.load_checkpoint(checkpoint_path)
    else:
        print("WARNING: Checkpoint not found. Evaluating untrained DQN.")
        
    stats = run_evaluation(agent, env, num_episodes=num_episodes, max_steps=env.max_steps)
    print(f"DQN -> Success Rate: {stats['success_rate']*100:.1f}%, Mean Steps: {stats['mean_steps']:.1f}")
    return stats

def main():
    os.makedirs("results", exist_ok=True)
    
    # 1. Evaluate classical baselines across all maps
    maps = ["office", "apartment", "school", "hospital", "mall"]
    baseline_results = {}
    
    for m in maps:
        baseline_results[m] = evaluate_baselines(m, num_episodes=10) # 10 is enough for deterministic algos
        
    with open("results/baseline_eval.json", "w") as f:
        json.dump(baseline_results, f, indent=4)
        
    print("\nSaved baseline evaluations to results/baseline_eval.json")
    
if __name__ == "__main__":
    main()
