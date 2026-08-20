"""
Environment package for Adaptive Evacuation GNN-DQN.

Public API:
    EvacuationEnv   — Gymnasium-compatible environment
    Grid            — 2D building grid
    CellType        — Cell type enum
    Action          — Agent action enum
    RewardConfig    — Reward constants dataclass
    EnvironmentState — Dynamic state manager
    Building        — Fire/smoke dynamics engine
"""

from environment.constants import Action, CellType, RewardConfig, ACTION_DELTAS, ACTION_NAMES
from environment.grid import Grid
from environment.state import EnvironmentState
from environment.building import Building
from environment.evacuation_env import EvacuationEnv

__all__ = [
    "EvacuationEnv",
    "Grid",
    "CellType",
    "Action",
    "RewardConfig",
    "ACTION_DELTAS",
    "ACTION_NAMES",
    "EnvironmentState",
    "Building",
]
