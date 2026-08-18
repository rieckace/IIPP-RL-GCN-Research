import os
import sys
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import load_config
from environment.evacuation_env import EvacuationEnv
from models.gnn.trainer import GNNDQNAgent

# Environments to check
ENVS_TO_CHECK = {
    "Apartment": {
        "config": "configs/gnn.yaml",
        "checkpoint": "checkpoints/gnn/best_model.pt"
    },
    "Office": {
        "config": "configs/gnn_office.yaml",
        "checkpoint": "checkpoints/gnn_office/best_model.pt"
    },
    "School": {
        "config": "configs/gnn_school.yaml",
        "checkpoint": "checkpoints/gnn_school/best_model.pt"
    },
    "Hospital": {
        "config": "configs/gnn_hospital.yaml",
        "checkpoint": "checkpoints/gnn_hospital/best_model.pt"
    },
    "Mall": {
        "config": "configs/gnn_mall.yaml",
        "checkpoint": "checkpoints/gnn_mall/best_model.pt"
    }
}

EPISODES = 100
SEEDS = list(range(42, 42 + EPISODES))

def get_env_config(agent_config_path):
    if not os.path.exists(agent_config_path):
        return None
    agent_cfg = load_config(agent_config_path, validate=False)
    env_config_path = agent_cfg.get("environment", {}).get("config_path", "configs/default.yaml")
    if not os.path.isabs(env_config_path):
        env_config_path = os.path.join(PROJECT_ROOT, env_config_path)
    if not os.path.exists(env_config_path):
        return None
    return load_config(env_config_path)

def sanity_check(env, agent, env_name):
    print(f"\n{'='*50}\nSANITY CHECK - {env_name}\n{'='*50}")
    obs, info = env.reset(seed=SEEDS[0])
    graph_obs = env.get_graph_observation()
    action = agent.act(graph_obs, explore=False)
    next_obs, reward, terminated, truncated, next_info = env.step(action)
    
    print(f"Environment: {env_name}")
    print(f"Model loaded successfully")
    print(f"Observation keys: {list(graph_obs.keys())}")
    print(f"Number of nodes: {graph_obs['node_features'].shape[0]}")
    print(f"Node feature dimension: {graph_obs['node_features'].shape[1]}")
    print(f"Action space: {env.action_space}")
    print(f"Initial agent position: {info['agent_position']}")
    
    exits = getattr(env, '_exits', [])
    print(f"Exit position(s): {exits}")
    
    print(f"Selected action: {action}")
    reason = next_info.get("reason", "N/A")
    if terminated or truncated:
        print(f"Termination reason: {reason}")
    else:
        print(f"Termination reason: Not terminated yet")
    print("="*50 + "\n")

def run_evaluation():
    results = []
    
    # Setup dirs
    out_dir = os.path.join(PROJECT_ROOT, "results", "metrics")
    fig_dir = os.path.join(PROJECT_ROOT, "results", "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    for env_name, paths in ENVS_TO_CHECK.items():
        chkpt_path = os.path.join(PROJECT_ROOT, paths["checkpoint"])
        config_path = os.path.join(PROJECT_ROOT, paths["config"])
        
        if not os.path.exists(chkpt_path):
            print(f"Environment: {env_name}\nCheckpoint unavailable — evaluation not performed.\n")
            results.append({
                "environment": env_name,
                "checkpoint": "N/A",
                "episodes": "N/A",
                "successful_episodes": "N/A",
                "failed_episodes": "N/A",
                "success_rate": "N/A",
                "avg_steps": "N/A",
                "avg_reward": "N/A",
                "total_hazard_events": "N/A",
                "avg_hazard_events": "N/A",
                "path_efficiency": "N/A",
                "seed": "N/A"
            })
            continue
            
        print(f"Evaluating {env_name}...")
        
        agent_cfg = load_config(config_path, validate=False)
        env_cfg = get_env_config(config_path)
        
        env = EvacuationEnv(env_cfg, render_mode=None)
        # Force random starts if not already true
        env.randomize_agent_start = True
        
        agent = GNNDQNAgent(agent_cfg)
        agent.load_checkpoint(chkpt_path)
        agent.epsilon = 0.0
        
        sanity_check(env, agent, env_name)
        
        successful_episodes = 0
        failed_episodes = 0
        total_steps = 0
        total_reward_all = 0.0
        total_hazard_events = 0
        
        for i in range(EPISODES):
            obs, info = env.reset(seed=SEEDS[i])
            episode_reward = 0.0
            steps = 0
            
            while True:
                graph_obs = env.get_graph_observation()
                with torch.inference_mode():
                    action = agent.act(graph_obs, explore=False)
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
                    
            reason = info.get("reason", "")
            if reason == "reached_exit":
                successful_episodes += 1
            else:
                failed_episodes += 1
                
            if reason in ["fire_caught_agent", "hit_fire"]:
                total_hazard_events += 1
                
            total_steps += steps
            total_reward_all += episode_reward
            
        success_rate = (successful_episodes / EPISODES) * 100
        avg_steps = total_steps / EPISODES
        avg_reward = total_reward_all / EPISODES
        avg_hazards = total_hazard_events / EPISODES
        
        res = {
            "environment": env_name,
            "checkpoint": paths["checkpoint"],
            "episodes": EPISODES,
            "successful_episodes": successful_episodes,
            "failed_episodes": failed_episodes,
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "total_hazard_events": total_hazard_events,
            "avg_hazard_events": avg_hazards,
            "path_efficiency": "N/A", # Not reliably computable without optimal A* path per random start
            "seed": f"{SEEDS[0]}-{SEEDS[-1]}"
        }
        results.append(res)
        
        print("==================================================")
        print(f"GCN-DQN EVALUATION RESULTS")
        print("==================================================")
        print(f"Environment: {env_name}")
        print(f"Checkpoint: {paths['checkpoint']}")
        print(f"Episodes: {EPISODES}")
        print(f"Successful Episodes: {successful_episodes}")
        print(f"Failed Episodes: {failed_episodes}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Steps: {avg_steps:.1f}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Total Hazard Events: {total_hazard_events}")
        print(f"Average Hazard Events: {avg_hazards:.2f}")
        print(f"Path Efficiency: N/A")
        print("--------------------------------------------------\n")

    # Save Results
    csv_path = os.path.join(out_dir, "evaluation_results.csv")
    json_path = os.path.join(out_dir, "evaluation_results.json")
    
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print("\nEnvironment | Success Rate | Avg Steps | Avg Reward | Hazard Events | Path Efficiency")
    for r in results:
        if r["success_rate"] == "N/A":
            print(f"{r['environment']:11} | N/A          | N/A       | N/A        | N/A           | N/A")
        else:
            print(f"{r['environment']:11} | {r['success_rate']:11.1f}% | {r['avg_steps']:<9.1f} | {r['avg_reward']:<10.2f} | {r['total_hazard_events']:<13} | {r['path_efficiency']}")
            
    # Plots
    valid_res = [r for r in results if r["success_rate"] != "N/A"]
    if valid_res:
        envs = [r["environment"] for r in valid_res]
        
        # 1. Success Rate
        plt.figure(figsize=(8, 6))
        sr = [r["success_rate"] for r in valid_res]
        plt.bar(envs, sr, color='green', alpha=0.7)
        plt.title('Evaluation Success Rate Comparison')
        plt.ylabel('Success Rate (%)')
        plt.ylim(0, 105)
        plt.savefig(os.path.join(fig_dir, "success_rate_comparison.png"), dpi=300)
        plt.close()
        
        # 2. Avg Steps
        plt.figure(figsize=(8, 6))
        steps = [r["avg_steps"] for r in valid_res]
        plt.bar(envs, steps, color='blue', alpha=0.7)
        plt.title('Evaluation Average Steps Comparison')
        plt.ylabel('Average Steps')
        plt.savefig(os.path.join(fig_dir, "average_steps_comparison.png"), dpi=300)
        plt.close()
        
        # 3. Avg Reward
        plt.figure(figsize=(8, 6))
        rews = [r["avg_reward"] for r in valid_res]
        plt.bar(envs, rews, color='purple', alpha=0.7)
        plt.title('Evaluation Average Reward Comparison')
        plt.ylabel('Average Reward')
        plt.savefig(os.path.join(fig_dir, "average_reward_comparison.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    run_evaluation()
