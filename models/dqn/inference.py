"""
inference.py

Functions for running episodes and evaluating a trained DQN agent.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from evaluation.statistics import success_rate_confidence_interval


def run_episode(
    env: Any,
    agent: Any,
    render: bool = False,
    explore: bool = False,
) -> Dict[str, Any]:
    """Run a single episode and collect metrics.

    Args:
        env:     Gymnasium-compatible EvacuationEnv instance.
        agent:   DQNAgent with an act() method.
        render:  If True, render each step (env must have render_mode set).
        explore: If True, use epsilon-greedy; if False, greedy only.

    Returns:
        Dict with keys: total_reward, steps, success, reason, fire_count.
    """
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        action = agent.act(obs, explore=explore)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    success = info.get("reason", "") == "reached_exit"

    return {
        "total_reward": total_reward,
        "steps": steps,
        "success": success,
        "reason": info.get("reason", "unknown"),
        "fire_count": info.get("fire_count", 0),
    }


def evaluate_agent(
    env: Any,
    agent: Any,
    num_episodes: int = 100,
    seed_start: int = 1000,
) -> Dict[str, Any]:
    """Evaluate an agent over multiple greedy episodes.

    Args:
        env:          EvacuationEnv instance.
        agent:        Trained DQNAgent.
        num_episodes: Number of evaluation episodes.
        seed_start:   Starting seed for reproducible evaluation.

    Returns:
        Dict with aggregated metrics:
            avg_reward, avg_steps, success_rate, outcomes (dict),
            all_rewards (list), all_steps (list).
    """
    all_rewards: List[float] = []
    all_steps: List[int] = []
    successes = 0
    outcomes: Dict[str, int] = {}

    for i in range(num_episodes):
        # Use deterministic seeds for reproducibility
        env.reset(seed=seed_start + i)
        result = run_episode(env, agent, explore=False)

        all_rewards.append(result["total_reward"])
        all_steps.append(result["steps"])
        if result["success"]:
            successes += 1

        reason = result["reason"]
        outcomes[reason] = outcomes.get(reason, 0) + 1

    success_rate = successes / num_episodes if num_episodes > 0 else 0.0
    ci_low, ci_high = success_rate_confidence_interval(successes, num_episodes)

    return {
        "avg_reward": np.mean(all_rewards),
        "std_reward": np.std(all_rewards),
        "avg_steps": np.mean(all_steps),
        "success_rate": success_rate,
        "success_count": successes,
        "failure_count": num_episodes - successes,
        "success_rate_ci95": (ci_low, ci_high),
        "num_episodes": num_episodes,
        "outcomes": outcomes,
        "all_rewards": all_rewards,
        "all_steps": all_steps,
    }
