"""
training_plots.py

Publication-quality matplotlib plots for DQN training curves.
Generates a 4-panel figure: reward, success rate, loss, and epsilon.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Use non-interactive backend when saving to file
matplotlib.use("Agg")


def _rolling_mean(data: List[float], window: int) -> np.ndarray:
    """Compute rolling mean with a given window size."""
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_training_curves(
    metrics_dict: Dict[str, List],
    save_path: str = "results/plots/dqn_training.png",
    window: int = 50,
    title: str = "DQN Training on Evacuation Environment",
) -> None:
    """Generate a 4-panel training curves figure.

    Panels:
        1. Top-left:     Episode reward (raw + rolling average)
        2. Top-right:    Success rate (rolling %)
        3. Bottom-left:  Training loss (rolling average)
        4. Bottom-right: Epsilon decay over episodes

    Args:
        metrics_dict: Dict from TrainingMetrics.to_dict() with keys:
                      reward, success, loss, epsilon.
        save_path:    Output file path for the saved figure.
        window:       Rolling average window size.
        title:        Overall figure title.
    """
    rewards = metrics_dict["reward"]
    successes = metrics_dict["success"]
    losses = metrics_dict["loss"]
    epsilons = metrics_dict["epsilon"]
    episodes = list(range(1, len(rewards) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    # --- Colour palette ---
    raw_color = "#4A90D9"
    avg_color = "#E8742C"
    success_color = "#2ECC71"
    loss_color = "#E74C3C"
    epsilon_color = "#9B59B6"

    # ── Panel 1: Reward ──
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards, alpha=0.25, color=raw_color, linewidth=0.8, label="Raw")
    if len(rewards) >= window:
        rolling = _rolling_mean(rewards, window)
        x_rolling = list(range(window, len(rewards) + 1))
        ax1.plot(x_rolling, rolling, color=avg_color, linewidth=2.0,
                 label=f"{window}-ep avg")
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Episode Reward")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Success Rate ──
    ax2 = axes[0, 1]
    if len(successes) >= window:
        rolling_sr = _rolling_mean(successes, window) * 100
        x_sr = list(range(window, len(successes) + 1))
        ax2.plot(x_sr, rolling_sr, color=success_color, linewidth=2.0)
        ax2.fill_between(x_sr, 0, rolling_sr, alpha=0.15, color=success_color)
    else:
        cumulative_sr = [np.mean(successes[:i+1]) * 100 for i in range(len(successes))]
        ax2.plot(episodes, cumulative_sr, color=success_color, linewidth=2.0)
    ax2.axhline(y=90, color="red", linestyle="--", linewidth=1.0,
                label="90% target", alpha=0.7)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_title(f"Success Rate ({window}-ep rolling)")
    ax2.set_ylim(-5, 105)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Loss ──
    ax3 = axes[1, 0]
    # Filter out zero losses (before learning starts)
    valid_losses = [(i + 1, l) for i, l in enumerate(losses) if l > 0]
    if valid_losses:
        loss_x, loss_y = zip(*valid_losses)
        ax3.plot(loss_x, loss_y, alpha=0.2, color=loss_color, linewidth=0.8, label="Raw")
        if len(loss_y) >= window:
            rolling_loss = _rolling_mean(list(loss_y), window)
            x_loss = list(range(loss_x[0] + window - 1,
                                loss_x[0] + window - 1 + len(rolling_loss)))
            ax3.plot(x_loss, rolling_loss, color=loss_color, linewidth=2.0,
                     label=f"{window}-ep avg")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Huber Loss")
    ax3.set_title("Training Loss")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Epsilon ──
    ax4 = axes[1, 1]
    ax4.plot(episodes, epsilons, color=epsilon_color, linewidth=2.0)
    ax4.fill_between(episodes, 0, epsilons, alpha=0.1, color=epsilon_color)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Epsilon (ε)")
    ax4.set_title("Exploration Rate Decay")
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(True, alpha=0.3)

    # ── Save ──
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Training plots saved to: {save_path}")
