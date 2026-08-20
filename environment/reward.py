"""
reward.py

Multi-objective reward function for the evacuation environment.
All reward magnitudes are controlled by a RewardConfig dataclass.
"""

from environment.heuristics import AStarPlanner
from environment.constants import CellType, RewardConfig
from environment.grid import Grid


def _get_exit_positions(grid: Grid) -> list[tuple[int, int]]:
    exits: list[tuple[int, int]] = []
    for row in range(grid.rows):
        for col in range(grid.cols):
            if grid.get_cell(row, col) == CellType.EXIT:
                exits.append((row, col))
    return exits


def compute_reward(
    old_pos: tuple[int, int],
    new_pos: tuple[int, int],
    grid: Grid,
    moved: bool,
    stayed: bool,
    reward_cfg: RewardConfig,
    dest_cell_override: int | None = None,
    visit_count: float = 1.0,
) -> tuple[float, bool, str]:
    """Compute the reward for a single transition.

    Args:
        old_pos:            Agent position before the step.
        new_pos:            Agent position after the step.
        grid:               Current grid (already has hazards placed for this step).
        moved:              Whether the agent actually changed position.
        stayed:             Whether the agent chose the STAY action.
        reward_cfg:         RewardConfig with all tuneable magnitudes.
        dest_cell_override: If provided, use this cell type instead of reading
                            from the grid. This is needed because the agent may
                            have already been placed on the grid, overwriting
                            the original cell (EXIT/FIRE).
        visit_count:        Number of times agent has stepped on new_pos in this episode.

    Returns:
        (reward, terminated, reason)
        - reward:     Scalar reward for this transition.
        - terminated: Whether the episode should end.
        - reason:     Human-readable explanation of the event.
    """
    if dest_cell_override is not None:
        cell = dest_cell_override
    else:
        cell = grid.get_cell(new_pos[0], new_pos[1])

    exits = _get_exit_positions(grid)

    # --- Terminal: agent reached an exit ---
    if cell == CellType.EXIT:
        return reward_cfg.exit_reached, True, "reached_exit"

    # --- Terminal: agent stepped into fire ---
    if cell == CellType.FIRE:
        return reward_cfg.fire_hit, True, "hit_fire"

    # --- Non-terminal: agent is standing in smoke ---
    if cell == CellType.SMOKE:
        reward = reward_cfg.smoke_step
        reason = "in_smoke"
    # --- Non-terminal: agent bumped into a wall (position unchanged) ---
    elif not moved and not stayed:
        return reward_cfg.wall_bump, False, "wall_bump"
    # --- Non-terminal: agent chose STAY ---
    elif stayed:
        return reward_cfg.stay_penalty, False, "stayed"
    # --- Non-terminal: normal valid step ---
    else:
        reward = reward_cfg.normal_step
        reason = "normal_step"

    # --- Count-Based Visited Penalty: penalize revisiting cells ---
    if moved and not stayed:
        if visit_count == 2.0:
            reward += -1.0
        elif visit_count >= 3.0:
            reward += -3.0

    # --- Dense shaping: reward moving closer to the nearest exit ---
    if exits and reward_cfg.exit_progress_scale != 0.0 and moved and not stayed:
        # Use actual path distance around walls for robust navigation
        old_path = AStarPlanner.compute_path(grid, old_pos, set(exits))
        new_path = AStarPlanner.compute_path(grid, new_pos, set(exits))
        
        old_distance = len(old_path) - 1 if old_path else min(
            AStarPlanner.manhattan_distance(old_pos, exit_pos) for exit_pos in exits
        )
        new_distance = len(new_path) - 1 if new_path else min(
            AStarPlanner.manhattan_distance(new_pos, exit_pos) for exit_pos in exits
        )
        reward += reward_cfg.exit_progress_scale * (old_distance - new_distance)

    return reward, False, reason
