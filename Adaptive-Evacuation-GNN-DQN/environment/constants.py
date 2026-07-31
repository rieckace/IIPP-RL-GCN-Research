"""
constants.py

Central definitions for cell types, actions, direction deltas,
and reward configuration used throughout the evacuation environment.
"""
 
from enum import IntEnum
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Cell types that can occupy a grid position
# ---------------------------------------------------------------------------
class CellType(IntEnum):
    EMPTY = 0
    WALL = 1
    AGENT = 2
    EXIT = 3
    OBSTACLE = 4
    SMOKE = 5
    FIRE = 6
    SENSOR = 7


# ---------------------------------------------------------------------------
# Discrete actions available to the agent
# ---------------------------------------------------------------------------
class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


# ---------------------------------------------------------------------------
# Direction deltas: maps each Action → (row_delta, col_delta)
# ---------------------------------------------------------------------------
ACTION_DELTAS: dict[Action, tuple[int, int]] = {
    Action.UP:    (-1,  0),
    Action.DOWN:  ( 1,  0),
    Action.LEFT:  ( 0, -1),
    Action.RIGHT: ( 0,  1),
    Action.STAY:  ( 0,  0),
}


# ---------------------------------------------------------------------------
# Human-readable labels (useful for logging / rendering)
# ---------------------------------------------------------------------------
ACTION_NAMES: dict[Action, str] = {
    Action.UP:    "UP",
    Action.DOWN:  "DOWN",
    Action.LEFT:  "LEFT",
    Action.RIGHT: "RIGHT",
    Action.STAY:  "STAY",
}


# ---------------------------------------------------------------------------
# Reward configuration — all tuneable reward magnitudes in one place
# ---------------------------------------------------------------------------
@dataclass
class RewardConfig:
    """Holds every reward constant so they can be overridden via config."""
    exit_reached: float = 100.0
    fire_hit: float = -50.0
    smoke_step: float = -10.0
    wall_bump: float = -5.0
    normal_step: float = -1.0
    stay_penalty: float = -2.0
    exit_progress_scale: float = 0.0
    team_bonus: float = 10.0