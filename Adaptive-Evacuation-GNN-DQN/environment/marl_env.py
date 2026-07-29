"""
marl_env.py

Multi-Agent Reinforcement Learning (MARL) wrapper/subclass for the evacuation environment.
"""

import random
from typing import Any, Dict, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environment.actions import apply_action, is_valid_move
from environment.constants import Action, CellType
from environment.evacuation_env import EvacuationEnv
from environment.reward import compute_reward
from environment.renderer import clear_screen, render_frame

class MARLEvacuationEnv(EvacuationEnv):
    """Multi-Agent Evacuation Environment.
    
    Supports multiple agents moving simultaneously.
    Action Space: MultiDiscrete (num_agents actions).
    Observation Space: Remains flat (or graph) but with multiple agent positions.
    """
    
    def __init__(self, config: Dict[str, Any], render_mode: str | None = None):
        super().__init__(config, render_mode)
        
        self.num_agents = len(self._agent_starts)
        # Action space: A list of discrete actions, one for each agent
        self.action_space = spaces.MultiDiscrete([len(Action)] * self.num_agents)
        
        # Track agent alive status: True if alive/evacuating, False if exited/dead
        self.active_agents = [True] * self.num_agents
        self.agent_rewards = [0.0] * self.num_agents

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.active_agents = [True] * self.num_agents
        self.agent_rewards = [0.0] * self.num_agents
        info["agent_rewards"] = list(self.agent_rewards)
        info["active_agents"] = list(self.active_agents)
        info["agent_positions"] = list(self.state.agent_positions)
        return obs, info

    def step(self, actions: List[int]) -> Tuple[np.ndarray, List[float], bool, bool, Dict[str, Any]]:
        self._step_count += 1
        
        rewards = [0.0] * self.num_agents
        reasons = [""] * self.num_agents
        
        # 1. Propose moves
        proposed_positions = []
        for i, agent_pos in enumerate(self.state.agent_positions):
            if not self.active_agents[i]:
                proposed_positions.append(agent_pos)
                continue
                
            action = actions[i]
            new_pos = apply_action(agent_pos, action)
            
            # Basic bound check and wall check
            if is_valid_move(self.grid, new_pos[0], new_pos[1]):
                proposed_positions.append(new_pos)
            else:
                proposed_positions.append(agent_pos)

        # 2. Resolve collisions (simple: if two agents target same empty cell, cancel both moves for safety/simplicity)
        from collections import Counter
        pos_counts = Counter(proposed_positions)
        
        for i in range(self.num_agents):
            if not self.active_agents[i]:
                continue
            
            target_pos = proposed_positions[i]
            # If collision with another agent, revert to original position
            if pos_counts[target_pos] > 1:
                proposed_positions[i] = self.state.agent_positions[i]

        # 3. Apply moves and compute rewards
        any_escaped = False
        
        for i in range(self.num_agents):
            if not self.active_agents[i]:
                continue
                
            old_pos = self.state.agent_positions[i]
            new_pos = proposed_positions[i]
            action = actions[i]
            
            moved = (old_pos != new_pos)
            stayed = (Action(action) == Action.STAY)
            
            dest_cell = self.grid.get_cell(new_pos[0], new_pos[1])
            
            reward, terminated, reason = compute_reward(
                old_pos=old_pos,
                new_pos=new_pos,
                grid=self.grid,
                moved=moved,
                stayed=stayed,
                reward_cfg=self.reward_config,
                dest_cell_override=dest_cell
            )
            
            if reason == "reached_exit":
                any_escaped = True
                self.active_agents[i] = False
                new_pos = (-1, -1) # Remove from grid
            
            rewards[i] = reward
            reasons[i] = reason
            self.state.agent_positions[i] = new_pos
            self.agent_rewards[i] += reward

        # Re-sync grid without fire/smoke (handled in sync_to_grid)
        self.state.sync_to_grid(self.grid)

        # 4. Advance hazards
        # Check if anyone is still active
        if any(self.active_agents):
            self.building.step(self.grid, self.state, self._rng)
            
            # Check fire hits
            for i in range(self.num_agents):
                if not self.active_agents[i]:
                    continue
                
                ar, ac = self.state.agent_positions[i]
                if (ar, ac) in self.state.fire_cells:
                    rewards[i] = self.reward_config.fire_hit
                    self.agent_rewards[i] += rewards[i]
                    reasons[i] = "hit_fire"
                    self.active_agents[i] = False
                    self.state.agent_positions[i] = (-1, -1)
                    
            self.state.sync_to_grid(self.grid)

        # 5. Apply Team Bonus
        if any_escaped:
            for i in range(self.num_agents):
                if self.active_agents[i]: # Only reward currently active agents for teammates escaping
                    rewards[i] += self.reward_config.team_bonus
                    self.agent_rewards[i] += self.reward_config.team_bonus

        # 6. Global Termination
        terminated = not any(self.active_agents)
        truncated = (not terminated) and (self._step_count >= self.max_steps)

        # Aggregate observation
        obs = self.state.to_observation(self.grid)
        
        info = {
            "step": self._step_count,
            "agent_positions": self.state.agent_positions,
            "active_agents": self.active_agents,
            "agent_rewards": self.agent_rewards,
            "reasons": reasons,
            "total_reward": sum(self.agent_rewards)
        }
        
        if self.render_mode == "human":
            clear_screen()
            print(render_frame(self.grid, self._step_count, sum(rewards), sum(self.agent_rewards), self.max_steps))

        return obs, rewards, terminated, truncated, info
