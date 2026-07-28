"""
compare_models.py

Reads training metrics CSV files from different models (e.g., DQN and GNN)
and generates overlaid comparative plots.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Compare RL model training metrics.")
    parser.add_argument("--csv1", type=str, required=True, help="Path to first CSV (e.g., DQN)")
    parser.add_argument("--name1", type=str, default="Baseline DQN", help="Label for first model")
    parser.add_argument("--csv2", type=str, required=True, help="Path to second CSV (e.g., GNN)")
    parser.add_argument("--name2", type=str, default="GNN-DQN", help="Label for second model")
    parser.add_argument("--save-dir", type=str, default="results/plots", help="Directory to save plots")
    parser.add_argument("--window", type=int, default=50, help="Rolling average window")
    return parser.parse_args()

def plot_comparison(df1, name1, df2, name2, window, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate rolling averages
    df1_roll = df1.rolling(window=window, min_periods=1).mean(numeric_only=True)
    df2_roll = df2.rolling(window=window, min_periods=1).mean(numeric_only=True)
    
    # 1. Success Rate Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(df1['episode'], df1_roll['success'] * 100, label=name1, color='blue', linewidth=2)
    plt.plot(df2['episode'], df2_roll['success'] * 100, label=name2, color='orange', linewidth=2)
    plt.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90% Threshold')
    
    plt.title(f"Success Rate Comparison (Rolling Avg: {window} eps)", fontsize=14, pad=15)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_success_rate.png"), dpi=150)
    plt.close()
    
    # 2. Reward Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(df1['episode'], df1_roll['reward'], label=name1, color='blue', linewidth=2)
    plt.plot(df2['episode'], df2_roll['reward'], label=name2, color='orange', linewidth=2)
    
    plt.title(f"Average Reward Comparison (Rolling Avg: {window} eps)", fontsize=14, pad=15)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_reward.png"), dpi=150)
    plt.close()
    
    print(f"[+] Comparative plots saved to {save_dir}/")

def main():
    args = parse_args()
    
    try:
        df1 = pd.read_csv(args.csv1)
        df2 = pd.read_csv(args.csv2)
    except FileNotFoundError as e:
        print(f"[!] Error loading CSV: {e}")
        return
        
    plot_comparison(df1, args.name1, df2, args.name2, args.window, args.save_dir)

if __name__ == "__main__":
    main()
