"""
test_environment.py

Unit tests for the Adaptive Evacuation environment.
Covers grid operations, actions, rewards, fire dynamics,
and full episode lifecycle.

Usage:
    cd Adaptive-Evacuation-GNN-DQN
    python -m pytest tests/test_environment.py -v
"""

import os
import sys
import random

import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from environment.constants import CellType, Action, ACTION_DELTAS, RewardConfig
from environment.grid import Grid
from environment.actions import apply_action, is_valid_move
from environment.reward import compute_reward
from environment.state import EnvironmentState
from environment.building import Building
from environment.evacuation_env import EvacuationEnv
from utils.config_loader import load_config


# ======================================================================
# Fixtures
# ======================================================================
@pytest.fixture
def small_config():
    """A minimal 5×5 config for fast testing."""
    return {
        "grid": {"rows": 5, "cols": 5},
        "map": {
            "walls": [[1, 1], [2, 2]],
            "exits": [[0, 4]],
            "fire_sources": [[3, 0]],
            "agent_start": [[0, 0]],
        },
        "dynamics": {
            "fire_spread_probability": 0.5,
            "smoke_radius": 1,
            "max_steps": 50,
        },
        "rewards": {
            "exit_reached": 100.0,
            "fire_hit": -50.0,
            "smoke_step": -10.0,
            "wall_bump": -5.0,
            "normal_step": -1.0,
            "stay_penalty": -2.0,
        },
    }


@pytest.fixture
def default_config():
    """Load the default YAML config."""
    config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")
    return load_config(config_path)


# ======================================================================
# Grid Tests
# ======================================================================
class TestGrid:
    def test_creation_dimensions(self):
        grid = Grid(10, 10)
        assert grid.rows == 10
        assert grid.cols == 10
        assert len(grid.grid) == 10
        assert len(grid.grid[0]) == 10

    def test_all_cells_empty_on_creation(self):
        grid = Grid(5, 5)
        for r in range(5):
            for c in range(5):
                assert grid.get_cell(r, c) == CellType.EMPTY

    def test_set_and_get_cell(self):
        grid = Grid(5, 5)
        grid.set_cell(2, 3, CellType.WALL)
        assert grid.get_cell(2, 3) == CellType.WALL

    def test_clear_cell(self):
        grid = Grid(5, 5)
        grid.set_cell(1, 1, CellType.FIRE)
        grid.clear_cell(1, 1)
        assert grid.get_cell(1, 1) == CellType.EMPTY

    def test_is_valid_position(self):
        grid = Grid(5, 5)
        assert grid.is_valid_position(0, 0) is True
        assert grid.is_valid_position(4, 4) is True
        assert grid.is_valid_position(-1, 0) is False
        assert grid.is_valid_position(0, 5) is False
        assert grid.is_valid_position(5, 5) is False

    def test_get_neighbors_corner(self):
        grid = Grid(5, 5)
        neighbors = grid.get_neighbors(0, 0)
        assert len(neighbors) == 2
        assert (1, 0) in neighbors
        assert (0, 1) in neighbors

    def test_get_neighbors_center(self):
        grid = Grid(5, 5)
        neighbors = grid.get_neighbors(2, 2)
        assert len(neighbors) == 4

    def test_to_numpy_shape(self):
        grid = Grid(10, 10)
        arr = grid.to_numpy()
        assert arr.shape == (10, 10)
        assert arr.dtype == np.int32

    def test_place_entities(self):
        grid = Grid(5, 5)
        grid.place_entities({
            "walls": [[0, 1], [1, 0]],
            "exits": [[4, 4]],
            "fire_sources": [[2, 2]],
        })
        assert grid.get_cell(0, 1) == CellType.WALL
        assert grid.get_cell(1, 0) == CellType.WALL
        assert grid.get_cell(4, 4) == CellType.EXIT
        assert grid.get_cell(2, 2) == CellType.FIRE

    def test_reset_clears_all(self):
        grid = Grid(5, 5)
        grid.set_cell(0, 0, CellType.WALL)
        grid.set_cell(3, 3, CellType.FIRE)
        grid.reset()
        for r in range(5):
            for c in range(5):
                assert grid.get_cell(r, c) == CellType.EMPTY


# ======================================================================
# Action Tests
# ======================================================================
class TestActions:
    def test_apply_action_up(self):
        new = apply_action((2, 3), Action.UP)
        assert new == (1, 3)

    def test_apply_action_down(self):
        new = apply_action((2, 3), Action.DOWN)
        assert new == (3, 3)

    def test_apply_action_left(self):
        new = apply_action((2, 3), Action.LEFT)
        assert new == (2, 2)

    def test_apply_action_right(self):
        new = apply_action((2, 3), Action.RIGHT)
        assert new == (2, 4)

    def test_apply_action_stay(self):
        new = apply_action((2, 3), Action.STAY)
        assert new == (2, 3)

    def test_is_valid_move_empty_cell(self):
        grid = Grid(5, 5)
        assert is_valid_move(grid, 2, 2) is True

    def test_is_valid_move_wall(self):
        grid = Grid(5, 5)
        grid.set_cell(2, 2, CellType.WALL)
        assert is_valid_move(grid, 2, 2) is False

    def test_is_valid_move_out_of_bounds(self):
        grid = Grid(5, 5)
        assert is_valid_move(grid, -1, 0) is False
        assert is_valid_move(grid, 5, 0) is False

    def test_is_valid_move_fire_is_valid(self):
        """Agent CAN step into fire (but gets negative reward)."""
        grid = Grid(5, 5)
        grid.set_cell(2, 2, CellType.FIRE)
        assert is_valid_move(grid, 2, 2) is True


# ======================================================================
# Reward Tests
# ======================================================================
class TestReward:
    def setup_method(self):
        self.cfg = RewardConfig()
        self.grid = Grid(5, 5)

    def test_exit_reached(self):
        self.grid.set_cell(0, 4, CellType.EXIT)
        reward, terminated, reason = compute_reward(
            (0, 3), (0, 4), self.grid, moved=True, stayed=False, reward_cfg=self.cfg
        )
        assert reward == 100.0
        assert terminated is True
        assert reason == "reached_exit"

    def test_fire_hit(self):
        self.grid.set_cell(2, 2, CellType.FIRE)
        reward, terminated, reason = compute_reward(
            (2, 1), (2, 2), self.grid, moved=True, stayed=False, reward_cfg=self.cfg
        )
        assert reward == -50.0
        assert terminated is True
        assert reason == "hit_fire"

    def test_smoke_step(self):
        self.grid.set_cell(1, 1, CellType.SMOKE)
        reward, terminated, reason = compute_reward(
            (1, 0), (1, 1), self.grid, moved=True, stayed=False, reward_cfg=self.cfg
        )
        assert reward == -10.0
        assert terminated is False

    def test_wall_bump(self):
        reward, terminated, reason = compute_reward(
            (0, 0), (0, 0), self.grid, moved=False, stayed=False, reward_cfg=self.cfg
        )
        assert reward == -5.0
        assert terminated is False
        assert reason == "wall_bump"

    def test_normal_step(self):
        reward, terminated, reason = compute_reward(
            (0, 0), (0, 1), self.grid, moved=True, stayed=False, reward_cfg=self.cfg
        )
        assert reward == -1.0
        assert terminated is False

    def test_stay_penalty(self):
        reward, terminated, reason = compute_reward(
            (1, 1), (1, 1), self.grid, moved=True, stayed=True, reward_cfg=self.cfg
        )
        assert reward == -2.0
        assert terminated is False


# ======================================================================
# State Tests
# ======================================================================
class TestState:
    def test_reset(self):
        state = EnvironmentState()
        state.reset(
            agent_starts=[[0, 0]],
            fire_sources=[[3, 3]],
            exit_positions=[[4, 4]],
            wall_positions=[[1, 1]],
        )
        assert state.agent_positions == [(0, 0)]
        assert (3, 3) in state.fire_cells
        assert (4, 4) in state.exit_cells
        assert (1, 1) in state.wall_cells

    def test_to_observation_shape(self):
        grid = Grid(5, 5)
        state = EnvironmentState()
        obs = state.to_observation(grid)
        assert obs.shape == (25,)
        assert obs.dtype == np.int32

    def test_to_graph_structure(self):
        grid = Grid(3, 3)
        state = EnvironmentState()
        graph = state.to_graph(grid)
        assert "node_features" in graph
        assert "edge_index" in graph
        assert graph["node_features"].shape == (9, len(CellType))
        assert graph["edge_index"].shape[0] == 2
        # 3×3 grid has 12 edges (each internal edge bidirectional)
        assert graph["edge_index"].shape[1] == 24  # 12 bidirectional pairs

    def test_sync_to_grid(self):
        grid = Grid(5, 5)
        grid.set_cell(4, 4, CellType.EXIT)
        state = EnvironmentState()
        state.agent_positions = [(0, 0)]
        state.fire_cells = {(3, 3)}
        state.smoke_cells = {(3, 2)}
        state.sync_to_grid(grid)
        assert grid.get_cell(0, 0) == CellType.AGENT
        assert grid.get_cell(3, 3) == CellType.FIRE
        assert grid.get_cell(3, 2) == CellType.SMOKE
        assert grid.get_cell(4, 4) == CellType.EXIT


# ======================================================================
# Building Dynamics Tests
# ======================================================================
class TestBuilding:
    def test_fire_spread_deterministic(self):
        """With probability 1.0, fire should always spread."""
        grid = Grid(5, 5)
        state = EnvironmentState()
        state.fire_cells = {(2, 2)}
        state.sync_to_grid(grid)

        building = Building(fire_spread_probability=1.0, smoke_radius=1)
        rng = random.Random(42)
        building.spread_fire(grid, state, rng)

        # All 4 neighbors of (2,2) should now be on fire
        for pos in [(1, 2), (3, 2), (2, 1), (2, 3)]:
            assert pos in state.fire_cells, f"{pos} should be on fire"

    def test_fire_no_spread_zero_prob(self):
        """With probability 0.0, fire should never spread."""
        grid = Grid(5, 5)
        state = EnvironmentState()
        state.fire_cells = {(2, 2)}
        state.sync_to_grid(grid)

        building = Building(fire_spread_probability=0.0, smoke_radius=1)
        rng = random.Random(42)
        building.spread_fire(grid, state, rng)

        assert len(state.fire_cells) == 1

    def test_fire_does_not_spread_to_walls(self):
        grid = Grid(5, 5)
        grid.set_cell(1, 2, CellType.WALL)
        state = EnvironmentState()
        state.fire_cells = {(2, 2)}
        state.wall_cells = {(1, 2)}
        state.sync_to_grid(grid)

        building = Building(fire_spread_probability=1.0, smoke_radius=1)
        rng = random.Random(42)
        building.spread_fire(grid, state, rng)

        assert (1, 2) not in state.fire_cells

    def test_smoke_around_fire(self):
        grid = Grid(5, 5)
        state = EnvironmentState()
        state.fire_cells = {(2, 2)}
        state.sync_to_grid(grid)

        building = Building(fire_spread_probability=0.0, smoke_radius=1)
        building.spread_smoke(grid, state)

        # Smoke should be in 4-connected neighbors
        for pos in [(1, 2), (3, 2), (2, 1), (2, 3)]:
            assert pos in state.smoke_cells, f"Smoke should be at {pos}"

    def test_smoke_not_on_fire_cell(self):
        grid = Grid(5, 5)
        state = EnvironmentState()
        state.fire_cells = {(2, 2)}
        state.sync_to_grid(grid)

        building = Building(fire_spread_probability=0.0, smoke_radius=2)
        building.spread_smoke(grid, state)

        assert (2, 2) not in state.smoke_cells


# ======================================================================
# Full Environment Tests
# ======================================================================
class TestEvacuationEnv:
    def test_reset_returns_correct_shape(self, small_config):
        env = EvacuationEnv(small_config)
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.int32

    def test_step_returns_correct_format(self, small_config):
        env = EvacuationEnv(small_config)
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(Action.RIGHT)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_space_size(self, small_config):
        env = EvacuationEnv(small_config)
        assert env.action_space.n == 5

    def test_episode_terminates_on_exit(self):
        """Place agent next to exit, move into it."""
        config = {
            "grid": {"rows": 3, "cols": 3},
            "map": {
                "walls": [],
                "exits": [[0, 2]],
                "fire_sources": [],
                "agent_start": [[0, 1]],
            },
            "dynamics": {
                "fire_spread_probability": 0.0,
                "smoke_radius": 0,
                "max_steps": 10,
            },
            "rewards": {
                "exit_reached": 100.0,
                "fire_hit": -50.0,
                "smoke_step": -10.0,
                "wall_bump": -5.0,
                "normal_step": -1.0,
                "stay_penalty": -2.0,
            },
        }
        env = EvacuationEnv(config)
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(Action.RIGHT)
        assert terminated is True
        assert reward == 100.0
        assert info["reason"] == "reached_exit"

    def test_episode_terminates_on_fire(self):
        """Place agent next to fire, move into it."""
        config = {
            "grid": {"rows": 3, "cols": 3},
            "map": {
                "walls": [],
                "exits": [[2, 2]],
                "fire_sources": [[0, 2]],
                "agent_start": [[0, 1]],
            },
            "dynamics": {
                "fire_spread_probability": 0.0,
                "smoke_radius": 0,
                "max_steps": 10,
            },
            "rewards": {
                "exit_reached": 100.0,
                "fire_hit": -50.0,
                "smoke_step": -10.0,
                "wall_bump": -5.0,
                "normal_step": -1.0,
                "stay_penalty": -2.0,
            },
        }
        env = EvacuationEnv(config)
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(Action.RIGHT)
        assert terminated is True
        assert reward == -50.0
        assert info["reason"] == "hit_fire"

    def test_max_steps_truncation(self):
        """Episode truncates after max_steps."""
        config = {
            "grid": {"rows": 3, "cols": 3},
            "map": {
                "walls": [],
                "exits": [[2, 2]],
                "fire_sources": [],
                "agent_start": [[0, 0]],
            },
            "dynamics": {
                "fire_spread_probability": 0.0,
                "smoke_radius": 0,
                "max_steps": 3,
            },
            "rewards": {
                "exit_reached": 100.0,
                "fire_hit": -50.0,
                "smoke_step": -10.0,
                "wall_bump": -5.0,
                "normal_step": -1.0,
                "stay_penalty": -2.0,
            },
        }
        env = EvacuationEnv(config)
        env.reset(seed=0)
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(Action.STAY)
        assert truncated is True
        assert info["reason"] == "max_steps_exceeded"

    def test_wall_bump_no_movement(self, small_config):
        """Agent should not move when walking into a wall."""
        env = EvacuationEnv(small_config)
        env.reset(seed=42)
        # Agent at (0,0), move UP → hits boundary
        obs, reward, terminated, truncated, info = env.step(Action.UP)
        assert info["agent_position"] == (0, 0)
        assert reward == -5.0

    def test_graph_observation_shape(self, small_config):
        env = EvacuationEnv(small_config)
        env.reset(seed=42)
        graph = env.get_graph_observation()
        num_nodes = small_config["grid"]["rows"] * small_config["grid"]["cols"]
        assert graph["node_features"].shape[0] == num_nodes
        assert graph["node_features"].shape[1] == len(CellType)
        assert graph["edge_index"].shape[0] == 2

    def test_observation_within_space(self, small_config):
        """Every observation value should be within observation_space bounds."""
        env = EvacuationEnv(small_config)
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        for _ in range(10):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break

    def test_default_config_loads(self, default_config):
        """Verify the default YAML config creates a valid env."""
        env = EvacuationEnv(default_config)
        obs, info = env.reset(seed=0)
        assert obs.shape == (100,)  # 10×10 grid


# ======================================================================
# Config Loader Tests
# ======================================================================
class TestConfigLoader:
    def test_load_default_config(self):
        config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")
        config = load_config(config_path)
        assert "grid" in config
        assert "map" in config
        assert "dynamics" in config
        assert "rewards" in config

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")
