import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from environment.make_env import make_env
from models.dqn.trainer import DQNAgent

def train_dqn(map_name="apartment", num_episodes=500, checkpoint_path="checkpoints/dqn/best_model.pt"):
    print(f"Starting DQN Training on Map: {map_name}")
    
    env = make_env(map_name)
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Use environment config from make_env
    config["environment"] = {"grid_rows": env.rows, "grid_cols": env.cols}
    
    # Increase batch size and memory for CNN
    config["agent"] = {
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,  # Faster decay for quick testing
        "learning_rate": 0.0005,
        "batch_size": 32,
        "target_update_freq": 20,
        "replay_buffer_size": 10000
    }
    
    agent = DQNAgent(config)
    
    best_reward = float('-inf')
    rewards = []
    
    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.act(obs, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            done = terminated or truncated
            
            # Store in replay buffer
            agent.memory.push(obs, action, reward, next_obs, done)
            
            # Train
            agent.learn()
            
            obs = next_obs
            episode_reward += reward
            
        # Update target network
        if ep % agent.target_update_freq == 0:
            agent.update_target()
            
        agent.decay_epsilon()
        rewards.append(episode_reward)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_checkpoint(checkpoint_path)
            
        avg = sum(rewards[-10:]) / len(rewards[-10:])
        print(f"Episode {ep}/{num_episodes} | Reward: {episode_reward:.1f} | Avg (Last 10): {avg:.1f} | Eps: {agent.epsilon:.3f}")
            
    print("Training Complete!")
    return agent

if __name__ == "__main__":
    os.makedirs("checkpoints/dqn", exist_ok=True)
    train_dqn(map_name="apartment", num_episodes=30)
