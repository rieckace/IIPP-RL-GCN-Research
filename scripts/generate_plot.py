import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(r"C:\Users\Acer\Downloads\Modified Plan for Research\Adaptive-Evacuation-GCN-DQN-Research-Clean-E0\results\research-results")
LEGACY_CSV = RESULTS_DIR / "gnn_evaluation_legacy.csv"
CORRECTED_CSV = RESULTS_DIR / "gnn_evaluation_corrected.csv"

def main():
    df_legacy = pd.read_csv(LEGACY_CSV)
    df_corrected = pd.read_csv(CORRECTED_CSV)
    
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

if __name__ == '__main__':
    main()
