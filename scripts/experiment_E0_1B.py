import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PROJECT_ROOT = Path(r"C:\Users\Acer\Downloads\Modified Plan for Research\Adaptive-Evacuation-GCN-DQN-Research-Clean-E0")
EVAL_SCRIPT = PROJECT_ROOT / "evaluation" / "run_evaluation.py"
RESULTS_DIR = PROJECT_ROOT / "results" / "research-results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LEGACY_CSV = RESULTS_DIR / "gnn_evaluation_legacy.csv"
CORRECTED_CSV = RESULTS_DIR / "gnn_evaluation_corrected.csv"
COMBINED_CSV = RESULTS_DIR / "gnn_evaluation_comparison.csv"

def run_evaluation(env_var, output_csv):
    env = os.environ.copy()
    env["LEGACY_GRAPH"] = env_var
    
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--output", str(output_csv),
        "--episodes", "100",
        "--seed-start", "2000"
    ]
    
    print(f"Running evaluation with LEGACY_GRAPH={env_var}...")
    subprocess.run(cmd, env=env, check=True)
    print(f"Done. Saved to {output_csv}\n")

def main():
    # Run Condition A (Legacy)
    run_evaluation("1", LEGACY_CSV)
    
    # Run Condition B (Corrected)
    run_evaluation("0", CORRECTED_CSV)
    
    # Combine and compare
    df_legacy = pd.read_csv(LEGACY_CSV)
    df_corrected = pd.read_csv(CORRECTED_CSV)
    
    df_legacy['condition'] = 'Legacy (Wall-Crossing)'
    df_corrected['condition'] = 'Corrected (Wall-Aware)'
    
    df_combined = pd.concat([df_legacy, df_corrected], ignore_index=True)
    df_combined.to_csv(COMBINED_CSV, index=False)
    
    # Print comparison table
    print("=== Comparison Table ===")
    for env in df_legacy['environment'].unique():
        print(f"\nEnvironment: {env.upper()}")
        print("-" * 60)
        
        row_leg = df_legacy[df_legacy['environment'] == env].iloc[0]
        row_cor = df_corrected[df_corrected['environment'] == env].iloc[0]
        
        print(f"{'Metric':<25} | {'Legacy':<15} | {'Corrected':<15} | {'Diff'}")
        print("-" * 60)
        
        # Success Rate
        sr_l = row_leg['success_rate'] * 100
        sr_c = row_cor['success_rate'] * 100
        print(f"{'Success Rate (%)':<25} | {sr_l:<15.1f} | {sr_c:<15.1f} | {sr_c - sr_l:+.1f}")
        
        # Avg Steps
        st_l = row_leg['avg_steps']
        st_c = row_cor['avg_steps']
        print(f"{'Avg Steps':<25} | {st_l:<15.2f} | {st_c:<15.2f} | {st_c - st_l:+.2f}")
        
        # Avg Reward
        rw_l = row_leg['avg_reward']
        rw_c = row_cor['avg_reward']
        print(f"{'Avg Reward':<25} | {rw_l:<15.2f} | {rw_c:<15.2f} | {rw_c - rw_l:+.2f}")
        
        # Hazard Events
        hz_l = row_leg['avg_hazard_events']
        hz_c = row_cor['avg_hazard_events']
        print(f"{'Avg Hazard Events':<25} | {hz_l:<15.2f} | {hz_c:<15.2f} | {hz_c - hz_l:+.2f}")
        
        # Path Efficiency
        pe_l = row_leg['path_efficiency'] * 100 if not pd.isna(row_leg['path_efficiency']) else 0.0
        pe_c = row_cor['path_efficiency'] * 100 if not pd.isna(row_cor['path_efficiency']) else 0.0
        print(f"{'Path Efficiency (%)':<25} | {pe_l:<15.1f} | {pe_c:<15.1f} | {pe_c - pe_l:+.1f}")
        
        # Success / Failed count
        print(f"{'Successful Episodes':<25} | {row_leg['successful_episodes']:<15} | {row_cor['successful_episodes']:<15} | ")
        print(f"{'Failed Episodes':<25} | {row_leg['failed_episodes']:<15} | {row_cor['failed_episodes']:<15} | ")
        
    print("\n")
    
    # Generate Plot
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(df_legacy['environment']))
        width = 0.35
        
        sr_leg = df_legacy['success_rate'] * 100
        sr_cor = df_corrected['success_rate'] * 100
        
        rects1 = ax.bar(x - width/2, sr_leg, width, label='Legacy (Wall-Crossing)', color='skyblue')
        rects2 = ax.bar(x + width/2, sr_cor, width, label='Corrected (Wall-Aware)', color='salmon')
        
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Zero-Retrain GCN Graph Topology Impact')
        ax.set_xticks(x)
        ax.set_xticklabels(df_legacy['environment'])
        ax.legend()
        
        plt.tight_layout()
        plot_path = RESULTS_DIR / "success_rate_comparison.png"
        plt.savefig(plot_path)
        print(f"Saved plot to {plot_path}")
    except Exception as e:
        print(f"Failed to generate plot: {e}")

if __name__ == '__main__':
    main()
