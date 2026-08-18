import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training(csv_path, out_path, window=50):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Calculate moving averages
    df['Reward_MA'] = df['reward'].rolling(window=window, min_periods=1).mean()
    df['Success_MA'] = df['success'].rolling(window=window, min_periods=1).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot Reward
    ax1.plot(df['episode'], df['reward'], alpha=0.3, color='blue', label='Episode Reward')
    ax1.plot(df['episode'], df['Reward_MA'], color='darkblue', linewidth=2, label=f'MA ({window})')
    ax1.set_title('Training Reward (Apartment Environment)')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot Success Rate
    ax2.plot(df['episode'], df['Success_MA'], color='green', linewidth=2, label=f'Success Rate MA ({window})')
    ax2.set_title('Evacuation Success Rate')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved plot to {out_path}")

if __name__ == "__main__":
    csv_file = "results/logs/gnn_training_metrics.csv"
    out_file = "results/plots/apartment_training_plot.png"
    plot_training(csv_file, out_file)
