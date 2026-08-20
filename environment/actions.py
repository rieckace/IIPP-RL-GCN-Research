"""
actions.py

Helper functions for applying agent actions and validating moves
within the evacuation grid environment.
"""

from typing import Tuple

from environment.constants import Action, ACTION_DELTAS, CellType
from environment.grid import Grid


def apply_action(
    position: Tuple[int, int],
    action: int,
) -> Tuple[int, int]:
    """Compute the new position after applying an action.

    Args:
        position: Current (row, col) of the agent.
        action:   Integer action id (maps to Action enum).

    Returns:
        (new_row, new_col) — the intended destination (may be invalid).
    """
    action_enum = Action(action)
    dr, dc = ACTION_DELTAS[action_enum]
    return (position[0] + dr, position[1] + dc)


def is_valid_move(grid: Grid, row: int, col: int) -> bool:
    """Check whether moving to (row, col) is allowed.

    A move is valid if:
      1. The position is within grid bounds.
      2. The cell is NOT a wall.

    Note: Stepping on fire is *valid* (the agent can walk into fire)
    but will receive a large negative reward and end the episode.

    Args:
        grid: The current Grid instance.
        row:  Target row.
        col:  Target column.

    Returns:
        True if the agent may move to (row, col).
    """
    if not grid.is_valid_position(row, col):
        return False
    cell = grid.get_cell(row, col)
    if cell == CellType.WALL:
        return False
    return True
