"""
evacuation_env.py

Gymnasium-compatible environment for adaptive building evacuation.

The agent must navigate a 2D grid building to reach an exit while
avoiding dynamically spreading fire and smoke hazards.
"""

import random
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.actions import apply_action, is_valid_move
from environment.building import Building
from environment.constants import Action, CellType, RewardConfig
from environment.grid import Grid
from environment.renderer import render_frame, clear_screen
from environment.reward import compute_reward
from environment.state import EnvironmentState


class EvacuationEnv(gym.Env):
    """Single-agent evacuation environment with dynamic fire/smoke.

    Observation:
        Flat integer vector of the grid (rows * cols), where each
        element is a CellType value (0–7).

    Actions:
        0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT, 4 = STAY

    Reward:
        Multi-objective — see reward.py and RewardConfig for details.

    Termination:
        - Agent reaches an exit cell  (success)
        - Agent steps into fire       (failure)

    Truncation:
        - Step count exceeds max_steps
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        config: Dict[str, Any],
        render_mode: Optional[str] = None,
    ) -> None:
        """
        Args:
            config:      Configuration dict (typically loaded from YAML).
            render_mode: "human" for animated terminal, "ansi" for string.
        """
        super().__init__()

        # --- Unpack config ---
        grid_cfg = config.get("grid", {})
        self.rows = grid_cfg.get("rows", 10)
        self.cols = grid_cfg.get("cols", 10)

        map_cfg = config.get("map", {})
        self._walls = map_cfg.get("walls", [])
        self._exits = map_cfg.get("exits", [])
        self._fire_sources = map_cfg.get("fire_sources", [])
        self._agent_starts = map_cfg.get("agent_start", [[0, 0]])

        dynamics_cfg = config.get("dynamics", {})
        self._fire_spread_prob = dynamics_cfg.get("fire_spread_probability", 0.3)
        self._smoke_radius = dynamics_cfg.get("smoke_radius", 2)
        self.max_steps = dynamics_cfg.get("max_steps", 200)

        reward_cfg = config.get("rewards", {})
        self.reward_config = RewardConfig(
            exit_reached=reward_cfg.get("exit_reached", 100.0),
            fire_hit=reward_cfg.get("fire_hit", -50.0),
            smoke_step=reward_cfg.get("smoke_step", -10.0),
            wall_bump=reward_cfg.get("wall_bump", -5.0),
            normal_step=reward_cfg.get("normal_step", -1.0),
            stay_penalty=reward_cfg.get("stay_penalty", -2.0),
            exit_progress_scale=reward_cfg.get("exit_progress_scale", 0.0),
            team_bonus=reward_cfg.get("team_bonus", 10.0),
        )

        # --- Gymnasium spaces ---
        self.observation_space = spaces.Box(
            low=0,
            high=max(int(ct) for ct in CellType),
            shape=(self.rows * self.cols,),
            dtype=np.int32,
        )
        self.action_space = spaces.Discrete(len(Action))

        # --- Internal components ---
        self.grid = Grid(self.rows, self.cols)
        self.state = EnvironmentState()
        self.building = Building(self._fire_spread_prob, self._smoke_radius)
        self.render_mode = render_mode

        # --- Episode bookkeeping ---
        self._step_count = 0
        self._total_reward = 0.0
        self._rng: Optional[random.Random] = None
        self.visit_counts = np.zeros((self.rows, self.cols), dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to its initial state.

        Args:
            seed:    Optional seed for reproducibility.
            options: Not used (reserved for Gymnasium compatibility).

        Returns:
            (observation, info)
        """
        super().reset(seed=seed)
        self._rng = random.Random(seed)

        # Reset grid
        self.grid.reset()

        # Place static entities (walls, exits) on the grid
        self.grid.place_entities({
            "walls": self._walls,
            "exits": self._exits,
            "fire_sources": self._fire_sources,
        })

        # Check if we should randomize start position (for training generalization)
        agent_starts = self._agent_starts
        if hasattr(self, "randomize_agent_start") and self.randomize_agent_start:
            empty_cells = []
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.grid.get_cell(r, c) == CellType.EMPTY:
                        empty_cells.append([r, c])
            if empty_cells:
                selected_start = self._rng.choice(empty_cells)
                agent_starts = [selected_start]

        # Reset dynamic state
        self.state.reset(
            agent_starts=agent_starts,
            fire_sources=self._fire_sources,
            exit_positions=self._exits,
            wall_positions=self._walls,
        )

        # Sync agent onto grid
        self.state.sync_to_grid(self.grid)

        # Reset counters and visit counts
        self._step_count = 0
        self._total_reward = 0.0
        self.visit_counts.fill(0.0)
        for pos in self.state.agent_positions:
            self.visit_counts[pos[0], pos[1]] = 1.0

        obs = self.state.to_observation(self.grid)
        info = self._build_info(reason="reset")

        if self.render_mode == "human":
            clear_screen()
            print(render_frame(
                self.grid, 0, 0.0, 0.0, self.max_steps, reason="reset"
            ))

        return obs, info

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.

        Order of operations:
          1. Move the agent.
          2. Compute reward (based on destination cell).
          3. Advance fire/smoke dynamics.
          4. Check if fire reached the agent's new position.
          5. Return (obs, reward, terminated, truncated, info).

        Args:
            action: Integer action id (0–4).

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self._step_count += 1

        # --- 1. Compute intended new position ---
        agent_pos = self.state.agent_positions[0]
        new_pos = apply_action(agent_pos, action)

        stayed = (Action(action) == Action.STAY)
        moved = True

        if is_valid_move(self.grid, new_pos[0], new_pos[1]):
            # Capture destination cell type BEFORE overwriting it
            dest_cell = self.grid.get_cell(new_pos[0], new_pos[1])
            # Clear old agent cell
            self.grid.clear_cell(agent_pos[0], agent_pos[1])
            self.state.agent_positions[0] = new_pos
        else:
            # Invalid move — agent stays in place
            dest_cell = self.grid.get_cell(agent_pos[0], agent_pos[1])
            new_pos = agent_pos
            moved = False

        # Increment visit counts for the new position
        self.visit_counts[new_pos[0], new_pos[1]] += 1.0

        # --- 2. Compute reward based on the ORIGINAL destination cell ---
        reward, terminated, reason = compute_reward(
            old_pos=agent_pos,
            new_pos=new_pos,
            grid=self.grid,
            moved=moved,
            stayed=stayed,
            reward_cfg=self.reward_config,
            dest_cell_override=dest_cell,
            visit_count=self.visit_counts[new_pos[0], new_pos[1]],
        )

        # Place agent on grid (after reward computed, so we don't
        # overwrite exit/fire cells before checking them)
        self.grid.set_cell(new_pos[0], new_pos[1], CellType.AGENT)

        # --- 3. Advance dynamics (fire/smoke spread) ---
        if not terminated:
            self.building.step(self.grid, self.state, self._rng)

            # --- 4. Check if fire reached agent after spread ---
            ar, ac = self.state.agent_positions[0]
            if (ar, ac) in self.state.fire_cells:
                reward = self.reward_config.fire_hit
                terminated = True
                reason = "fire_caught_agent"

        # --- 5. Truncation check ---
        truncated = False
        if not terminated and self._step_count >= self.max_steps:
            truncated = True
            reason = "max_steps_exceeded"

        self._total_reward += reward
        obs = self.state.to_observation(self.grid)
        info = self._build_info(reason=reason, action=action, reward=reward)

        # --- Render ---
        if self.render_mode == "human":
            clear_screen()
            print(render_frame(
                self.grid,
                self._step_count,
                reward,
                self._total_reward,
                self.max_steps,
                action_taken=action,
                reason=reason,
            ))

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """Return an ANSI-rendered frame (for render_mode='ansi')."""
        if self.render_mode == "ansi":
            return render_frame(
                self.grid,
                self._step_count,
                0.0,
                self._total_reward,
                self.max_steps,
            )
        return None

    # ------------------------------------------------------------------
    # Graph observation (for GNN-based agents in later phases)
    # ------------------------------------------------------------------
    def get_graph_observation(self) -> Dict[str, np.ndarray]:
        """Return the current state as a graph dict.

        Returns:
            {"node_features": ndarray, "edge_index": ndarray}
        """
        return self.state.to_graph(self.grid, self.visit_counts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_info(
        self,
        reason: str = "",
        action: Optional[int] = None,
        reward: float = 0.0,
    ) -> Dict[str, Any]:
        """Build the info dict returned by reset() and step()."""
        return {
            "step": self._step_count,
            "agent_position": self.state.agent_positions[0],
            "fire_count": len(self.state.fire_cells),
            "smoke_count": len(self.state.smoke_cells),
            "total_reward": self._total_reward,
            "reason": reason,
        }
