import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import deque
from environment.marl_env import MARLEvacuationEnv
from environment.wrappers import MARLGraphObservationWrapper
from models.marl.trainer import MARLAgent
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("marl_training")

def _coord_to_id(r, c, cols):
    if r == -1: return -1
    return r * cols + c

def train(config: dict, num_episodes: int = None, seed: int = None):
    episodes = num_episodes or config.get("training", {}).get("num_episodes", 1500)
    seed = seed or config.get("environment", {}).get("seed", 42)

    base_env = MARLEvacuationEnv(config)
    env = MARLGraphObservationWrapper(base_env)
    agent = MARLAgent(config)

    metrics = {
        "episode": [],
        "reward": [],
        "success_rate": [],
        "epsilon": []
    }
    recent_successes = deque(maxlen=50)

    logger.info("Starting MARL GNN Training...")
    start_time = time.time()

    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=seed + ep)
        
        # Determine agent IDs from coordinates
        cols = base_env.cols
        agent_node_ids = [_coord_to_id(r, c, cols) for (r, c) in info["agent_positions"]]
        active_agents = info["active_agents"]

        while True:
            actions = agent.act(obs, active_agents, agent_node_ids)
            
            next_obs, rewards, terminated, truncated, next_info = env.step(actions)
            
            # Map next positions to IDs
            next_agent_node_ids = [_coord_to_id(r, c, cols) for (r, c) in next_info["agent_positions"]]
            
            # Calculate dones per agent
            dones = []
            for i in range(base_env.num_agents):
                # An agent is 'done' if they were active, but now are not
                dones.append(active_agents[i] and not next_info["active_agents"][i])
            
            agent.step(
                obs=obs,
                next_obs=next_obs,
                active_agents=active_agents,
                agent_node_ids=agent_node_ids,
                actions=actions,
                rewards=rewards,
                next_agent_node_ids=next_agent_node_ids,
                dones=dones
            )
            
            obs = next_obs
            active_agents = next_info["active_agents"]
            agent_node_ids = next_agent_node_ids
            
            if terminated or truncated:
                break

        agent.decay_epsilon()
        
        total_team_reward = next_info["total_reward"]
        
        # Did ALL agents escape?
        all_escaped = all(r == "reached_exit" for r in next_info["reasons"])
        recent_successes.append(1.0 if all_escaped else 0.0)
        
        sr = np.mean(recent_successes) * 100

        metrics["episode"].append(ep)
        metrics["reward"].append(total_team_reward)
        metrics["success_rate"].append(sr)
        metrics["epsilon"].append(agent.epsilon)

        if ep % 10 == 0:
            logger.info(f"Episode {ep:4d} | Team Reward: {total_team_reward:6.1f} | SR: {sr:5.1f}% | e: {agent.epsilon:.3f}")

        # Early Stopping
        if len(recent_successes) == 50 and sr >= 90.0:
            logger.info(f"[OK] Early Stop at Episode {ep}! Success rate: {sr}%")
            break

    os.makedirs("results/logs", exist_ok=True)
    df = pd.DataFrame(metrics)
    df.to_csv("results/logs/marl_training.csv", index=False)
    
    plt.figure(figsize=(10, 5))
    plt.plot(metrics["episode"], metrics["success_rate"], color='purple', label="Team Success Rate (%)")
    plt.axhline(90, color="red", linestyle="--", label="Target (90%)")
    plt.legend()
    plt.grid()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/marl_success_rate.png")
    
    env.close()
