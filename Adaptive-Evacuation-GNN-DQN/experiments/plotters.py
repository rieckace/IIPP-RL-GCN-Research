import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_baseline_comparison():
    """Plot success rates and steps for baselines across different maps."""
    if not os.path.exists("results/baseline_eval.json"):
        print("Error: results/baseline_eval.json not found.")
        return
        
    with open("results/baseline_eval.json", "r") as f:
        data = json.load(f)
        
    maps = list(data.keys())
    agents = list(data[maps[0]].keys())
    
    # Plot Success Rates
    x = np.arange(len(maps))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, agent in enumerate(agents):
        success_rates = [data[m][agent]["success_rate"] * 100 for m in maps]
        ax.bar(x + (i - 1.5) * width, success_rates, width, label=agent)
        
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Classical Baselines Success Rate by Building Complexity')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in maps])
    ax.legend()
    
    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/baseline_success_rates.png", dpi=300)
    print("Saved baseline_success_rates.png")

    # Plot Steps (Evacuation Time) for successful runs
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, agent in enumerate(agents):
        # We only care about steps if they actually found the exit, but for simplicity
        # we'll just plot mean steps here.
        mean_steps = [data[m][agent]["mean_steps"] for m in maps]
        ax.bar(x + (i - 1.5) * width, mean_steps, width, label=agent)
        
    ax.set_ylabel('Mean Evacuation Steps')
    ax.set_title('Classical Baselines Evacuation Time by Building Complexity')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in maps])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("results/plots/baseline_mean_steps.png", dpi=300)
    print("Saved baseline_mean_steps.png")

if __name__ == "__main__":
    plot_baseline_comparison()
