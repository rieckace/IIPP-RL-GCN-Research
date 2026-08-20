"""
metrics.py

Training metrics tracker for DQN experiments.
Records per-episode data and provides rolling averages and CSV export.
"""

import csv
import os
from typing import Any, Dict, List, Optional

import numpy as np


class TrainingMetrics:
    """Tracks and aggregates training metrics across episodes.

    Records: reward, steps, success, loss, epsilon, fire_count, reason.
    Provides rolling averages and CSV persistence.
    """

    def __init__(self) -> None:
        self.rewards: List[float] = []
        self.steps: List[int] = []
        self.successes: List[bool] = []
        self.losses: List[float] = []
        self.epsilons: List[float] = []
        self.fire_counts: List[int] = []
        self.reasons: List[str] = []

    def record(
        self,
        reward: float,
        steps: int,
        success: bool,
        loss: float,
        epsilon: float,
        fire_count: int = 0,
        reason: str = "",
    ) -> None:
        """Record metrics for one episode."""
        self.rewards.append(reward)
        self.steps.append(steps)
        self.successes.append(success)
        self.losses.append(loss)
        self.epsilons.append(epsilon)
        self.fire_counts.append(fire_count)
        self.reasons.append(reason)

    def rolling_average(self, window: int = 50) -> Dict[str, float]:
        """Compute rolling averages over the last `window` episodes.

        Returns:
            Dict with avg_reward, avg_steps, success_rate, avg_loss.
        """
        if len(self.rewards) == 0:
            return {"avg_reward": 0.0, "avg_steps": 0, "success_rate": 0.0, "avg_loss": 0.0}

        n = min(window, len(self.rewards))
        return {
            "avg_reward": float(np.mean(self.rewards[-n:])),
            "avg_steps": float(np.mean(self.steps[-n:])),
            "success_rate": float(np.mean(self.successes[-n:])),
            "avg_loss": float(np.mean([l for l in self.losses[-n:] if l > 0]) or 0.0),
        }

    def to_dict(self) -> Dict[str, List]:
        """Return all metrics as a dict of lists (for plotting)."""
        return {
            "reward": self.rewards,
            "steps": self.steps,
            "success": [int(s) for s in self.successes],
            "loss": self.losses,
            "epsilon": self.epsilons,
            "fire_count": self.fire_counts,
            "reason": self.reasons,
        }

    def save_csv(self, path: str) -> None:
        """Save all episode metrics to a CSV file.

        Args:
            path: Output file path.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "reward", "steps", "success",
                "loss", "epsilon", "fire_count", "reason",
            ])
            for i in range(len(self.rewards)):
                writer.writerow([
                    i + 1,
                    f"{self.rewards[i]:.2f}",
                    self.steps[i],
                    int(self.successes[i]),
                    f"{self.losses[i]:.6f}",
                    f"{self.epsilons[i]:.4f}",
                    self.fire_counts[i],
                    self.reasons[i],
                ])

    @classmethod
    def load_csv(cls, path: str) -> "TrainingMetrics":
        """Load metrics from a CSV file.

        Args:
            path: Input CSV file path.

        Returns:
            Populated TrainingMetrics instance.
        """
        metrics = cls()
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metrics.record(
                    reward=float(row["reward"]),
                    steps=int(row["steps"]),
                    success=bool(int(row["success"])),
                    loss=float(row["loss"]),
                    epsilon=float(row["epsilon"]),
                    fire_count=int(row["fire_count"]),
                    reason=row["reason"],
                )
        return metrics
