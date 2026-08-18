import os
import sys
import json
import csv
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to system path
PROJECT_ROOT = r"c:\Users\Public\Downloads\IIPP\IIPP-RL-GCN-Research-main\IIPP-RL-GCN-Research-main\Adaptive-Evacuation-GNN-DQN"
sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import load_config
from environment.make_env import make_env
from environment.wrappers import GraphObservationWrapper
from environment.constants import CellType
from environment.heuristics import AStarPlanner
from models.gnn.trainer import GNNDQNAgent

def run_sanity_check(env, agent, map_name, ckpt_path):
    print(f"\n==================================================")
    print(f"  SANITY CHECK: {map_name.upper()}")
    print(f"==================================================")
    print(f"  Environment:            {map_name}")
    print(f"  Checkpoint:             {ckpt_path}")
    print(f"  Model loaded:           [OK]")
    
    # Run a single episode to print stats
    obs, info = env.reset(seed=42)
    start_pos = env.unwrapped.state.agent_positions[0]
    
    exits = []
    for r in range(env.unwrapped.grid.rows):
        for c in range(env.unwrapped.grid.cols):
            if env.unwrapped.grid.get_cell(r, c) == CellType.EXIT:
                exits.append((r, c))
    
    print(f"  Observation shape:      {obs['node_features'].shape}")
    print(f"  Number of nodes:        {obs['node_features'].shape[0]}")
    print(f"  Node feature dim:       {obs['node_features'].shape[1]}")
    print(f"  Action space:           {env.action_space}")
    print(f"  Initial position:       {start_pos}")
    print(f"  Exit positions:         {exits}")
    
    selected_actions = []
    step_count = 0
    while True:
        action = agent.act(obs, explore=False)
        selected_actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        if terminated or truncated:
            break
            
    print(f"  Selected actions (first 10): {selected_actions[:10]}")
    print(f"  Steps taken:            {step_count}")
    print(f"  Termination reason:     {info.get('reason', 'unknown')}")
    print(f"==================================================\n")

def evaluate_map(map_name: str, episodes_count: int = 100):
    config_path = os.path.join(PROJECT_ROOT, "configs", "gnn.yaml")
    gnn_config = load_config(config_path, validate=False)
    
    # Initialize environment and wrap with GNN graph wrapper
    base_env = make_env(map_name)
    base_env.randomize_agent_start = True # Enable start state randomization
    env = GraphObservationWrapper(base_env)
    
    # Initialize agent and load checkpoint
    agent = GNNDQNAgent(gnn_config)
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "gnn", "best_model.pt")
    
    if not os.path.exists(ckpt_path):
        print(f"\n[!] Checkpoint unavailable for {map_name} — evaluation not performed.")
        return None
        
    agent.load_checkpoint(ckpt_path)
    agent.epsilon = 0.0 # Greedy action selection
    agent.q_network.eval() # inference mode
    
    # Run 1 sanity check first
    run_sanity_check(env, agent, map_name, ckpt_path)
    
    # Evaluation metrics collectors
    success_count = 0
    total_steps = []
    successful_steps = []
    total_rewards = []
    total_hazard_events = 0
    path_efficiencies = []
    
    seeds_used = list(range(2000, 2000 + episodes_count))
    
    for ep in range(episodes_count):
        # Controlled seed setting
        seed_val = seeds_used[ep]
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        
        obs, info = env.reset(seed=seed_val)
        start_pos = base_env.state.agent_positions[0]
        
        # Get exit locations to calculate start optimal path length
        exits = set()
        for r in range(base_env.grid.rows):
            for c in range(base_env.grid.cols):
                if base_env.grid.get_cell(r, c) == CellType.EXIT:
                    exits.add((r, c))
                    
        optimal_path = AStarPlanner.compute_path(base_env.grid, start_pos, exits)
        optimal_path_len = len(optimal_path) - 1 if optimal_path else 0
        
        ep_reward = 0.0
        step_count = 0
        ep_hazards = 0
        
        while True:
            with torch.no_grad():
                action = agent.act(obs, explore=False)
            obs, reward, terminated, truncated, info = env.step(action)
            
            ep_reward += reward
            step_count += 1
            
            # Count hazard events (smoke steps + fire catch)
            reason = info.get("reason", "")
            if reason == "in_smoke":
                ep_hazards += 1
            elif reason in ("hit_fire", "fire_caught_agent"):
                ep_hazards += 1
                
            if terminated or truncated:
                break
                
        total_steps.append(step_count)
        total_rewards.append(ep_reward)
        total_hazard_events += ep_hazards
        
        final_reason = info.get("reason", "")
        if final_reason == "reached_exit":
            success_count += 1
            successful_steps.append(step_count)
            if optimal_path_len > 0:
                pe = optimal_path_len / step_count
                path_efficiencies.append(pe)
                
    avg_steps_all = np.mean(total_steps)
    avg_steps_success = np.mean(successful_steps) if successful_steps else 0.0
    avg_reward = np.mean(total_rewards)
    success_rate = (success_count / episodes_count) * 100
    avg_hazards = total_hazard_events / episodes_count
    avg_path_efficiency = np.mean(path_efficiencies) if path_efficiencies else "N/A"
    
    result = {
        "environment": map_name,
        "checkpoint": os.path.basename(ckpt_path),
        "episodes": episodes_count,
        "successful_episodes": success_count,
        "failed_episodes": episodes_count - success_count,
        "success_rate": success_rate,
        "avg_steps": float(avg_steps_all),
        "avg_steps_success": float(avg_steps_success),
        "avg_reward": float(avg_reward),
        "total_hazard_events": total_hazard_events,
        "avg_hazard_events": float(avg_hazards),
        "path_efficiency": avg_path_efficiency,
        "seed_range": f"2000-{2000+episodes_count-1}"
    }
    
    return result

def save_and_plot(results):
    metrics_dir = os.path.join(PROJECT_ROOT, "results", "metrics")
    figures_dir = os.path.join(PROJECT_ROOT, "results", "figures")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(metrics_dir, "evaluation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "environment", "checkpoint", "episodes", "successful_episodes",
            "failed_episodes", "success_rate", "avg_steps", "avg_steps_success",
            "avg_reward", "total_hazard_events", "avg_hazard_events", "path_efficiency", "seed"
        ])
        for r in results:
            writer.writerow([
                r["environment"], r["checkpoint"], r["episodes"], r["successful_episodes"],
                r["failed_episodes"], f"{r['success_rate']:.1f}%", f"{r['avg_steps']:.2f}",
                f"{r['avg_steps_success']:.2f}", f"{r['avg_reward']:.2f}", r["total_hazard_events"],
                f"{r['avg_hazard_events']:.2f}", 
                f"{r['path_efficiency']:.4f}" if isinstance(r['path_efficiency'], float) else "N/A",
                r["seed_range"]
            ])
            
    # Save JSON
    json_path = os.path.join(metrics_dir, "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[+] Saved evaluation metrics to:")
    print(f"    CSV:  {csv_path}")
    print(f"    JSON: {json_path}")
    
    # Generate Plots
    envs = [r["environment"] for r in results]
    success_rates = [r["success_rate"] for r in results]
    avg_steps = [r["avg_steps"] for r in results]
    avg_rewards = [r["avg_reward"] for r in results]
    
    # 1. Success Rate Plot
    plt.figure(figsize=(8, 5))
    bars = plt.bar(envs, success_rates, color="#2ECC71", width=0.5, edgecolor="black")
    plt.title("GCN-DQN Success Rate by Environment", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Environment", fontsize=12)
    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 2, f"{height:.1f}%", ha='center', fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "success_rate_comparison.png"), dpi=150)
    plt.close()

    # 2. Average Steps Plot
    plt.figure(figsize=(8, 5))
    bars = plt.bar(envs, avg_steps, color="#3498DB", width=0.5, edgecolor="black")
    plt.title("GCN-DQN Average Evacuation Steps", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Environment", fontsize=12)
    plt.ylabel("Steps", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + max(avg_steps)*0.02, f"{height:.1f}", ha='center', fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "average_steps_comparison.png"), dpi=150)
    plt.close()

    # 3. Average Reward Plot
    plt.figure(figsize=(8, 5))
    bars = plt.bar(envs, avg_rewards, color="#E67E22", width=0.5, edgecolor="black")
    plt.title("GCN-DQN Average Cumulative Reward", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Environment", fontsize=12)
    plt.ylabel("Cumulative Reward", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + (5 if height >= 0 else -15), f"{height:+.1f}", ha='center', fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "average_reward_comparison.png"), dpi=150)
    plt.close()

    # 4. Path Efficiency Plot
    pe_vals = []
    pe_envs = []
    for r in results:
        if isinstance(r["path_efficiency"], float):
            pe_vals.append(r["path_efficiency"] * 100)
            pe_envs.append(r["environment"])
            
    if pe_vals:
        plt.figure(figsize=(8, 5))
        bars = plt.bar(pe_envs, pe_vals, color="#9B59B6", width=0.5, edgecolor="black")
        plt.title("GCN-DQN Path Efficiency (Successful Runs)", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Environment", fontsize=12)
        plt.ylabel("Path Efficiency (%)", fontsize=12)
        plt.ylim(0, 105)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, height + 2, f"{height:.1f}%", ha='center', fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "path_efficiency_comparison.png"), dpi=150)
        plt.close()
        
    print(f"[+] Saved comparison charts to: {figures_dir}")

def main():
    environments = ["office", "apartment", "school", "hospital", "mall"]
    results = []
    
    for m in environments:
        res = evaluate_map(m, episodes_count=100)
        if res:
            results.append(res)
            
    if not results:
        print("[!] No environments could be evaluated.")
        return
        
    # Print clean summary
    print("\n" + "=" * 60)
    print("GCN-DQN EVALUATION RESULTS")
    print("=" * 60)
    for r in results:
        print(f"Environment:         {r['environment'].capitalize()}")
        print(f"Checkpoint:          {r['checkpoint']}")
        print(f"Episodes:            {r['episodes']}")
        print(f"Successful Episodes: {r['successful_episodes']}")
        print(f"Failed Episodes:     {r['failed_episodes']}")
        print(f"Success Rate:        {r['success_rate']:.1f}%")
        print(f"Average Steps (All): {r['avg_steps']:.2f}")
        print(f"Average Steps (Succ):{r['avg_steps_success']:.2f}")
        print(f"Average Reward:      {r['avg_reward']:.2f}")
        print(f"Total Hazard Events: {r['total_hazard_events']}")
        print(f"Average Hazards/Ep:  {r['avg_hazard_events']:.2f}")
        pe_str = f"{r['path_efficiency']*100:.2f}%" if isinstance(r['path_efficiency'], float) else "N/A"
        print(f"Path Efficiency:     {pe_str}")
        print("-" * 60)
        
    # Print Table
    print("\n" + "=" * 80)
    print(f"{'Environment':<12} | {'Success Rate':<12} | {'Avg Steps':<10} | {'Avg Reward':<10} | {'Hazard Events':<13} | {'Path Efficiency':<15}")
    print("=" * 80)
    for r in results:
        pe_str = f"{r['path_efficiency']*100:.1f}%" if isinstance(r['path_efficiency'], float) else "N/A"
        print(f"{r['environment'].capitalize():<12} | {r['success_rate']:>10.1f}% | {r['avg_steps']:>9.2f} | {r['avg_reward']:>9.1f} | {r['total_hazard_events']:>13} | {pe_str:>15}")
    print("=" * 80 + "\n")
    
    save_and_plot(results)

if __name__ == "__main__":
    main()
