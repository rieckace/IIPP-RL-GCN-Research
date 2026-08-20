"""
renderer.py

Terminal-based renderer for the evacuation environment.
Uses ANSI colour codes and Unicode symbols for a clear,
colourful grid display.
"""

import os
from environment.constants import CellType, ACTION_NAMES
from environment.grid import Grid


# ---------------------------------------------------------------------------
# ANSI colour codes
# ---------------------------------------------------------------------------
class _Colors:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GRAY    = "\033[90m"
    BG_RED  = "\033[41m"
    BG_GRAY = "\033[100m"


# ---------------------------------------------------------------------------
# Cell → coloured symbol mapping
# ---------------------------------------------------------------------------
_CELL_DISPLAY = {
    CellType.EMPTY:    (_Colors.GRAY,    "·"),
    CellType.WALL:     (_Colors.WHITE,   "█"),
    CellType.AGENT:    (_Colors.BLUE,    "A"),
    CellType.EXIT:     (_Colors.GREEN,   "E"),
    CellType.OBSTACLE: (_Colors.YELLOW,  "▓"),
    CellType.SMOKE:    (_Colors.MAGENTA, "░"),
    CellType.FIRE:     (_Colors.RED,     "F"),
    CellType.SENSOR:   (_Colors.CYAN,    "S"),
}


def render_frame(
    grid: Grid,
    step_num: int,
    reward: float,
    total_reward: float,
    max_steps: int,
    action_taken: int | None = None,
    reason: str = "",
) -> str:
    """Render the current grid state as a coloured ANSI string.

    Args:
        grid:         The Grid to render.
        step_num:     Current timestep number.
        reward:       Reward received this step.
        total_reward: Cumulative reward so far.
        max_steps:    Maximum allowed steps (for display).
        action_taken: Action id taken this step (or None on reset).
        reason:       Human-readable event description.

    Returns:
        Multi-line string suitable for printing to the terminal.
    """
    C = _Colors
    lines: list[str] = []

    # --- Header ---
    lines.append("")
    lines.append(f"{C.BOLD}{C.CYAN}╔{'═' * (grid.cols * 2 + 1)}╗{C.RESET}")

    # --- Grid rows ---
    for row in grid.grid:
        cells = []
        for cell in row:
            colour, symbol = _CELL_DISPLAY.get(
                cell, (_Colors.GRAY, "?")
            )
            cells.append(f"{colour}{C.BOLD}{symbol}{C.RESET}")
        line = " ".join(cells)
        lines.append(f"{C.CYAN}║{C.RESET} {line} {C.CYAN}║{C.RESET}")

    # --- Footer ---
    lines.append(f"{C.BOLD}{C.CYAN}╚{'═' * (grid.cols * 2 + 1)}╝{C.RESET}")

    # --- Info bar ---
    action_str = ACTION_NAMES.get(action_taken, "—") if action_taken is not None else "—"
    lines.append(
        f"  Step: {C.BOLD}{step_num:>3}{C.RESET}/{max_steps}"
        f"  |  Action: {C.BOLD}{action_str:<5}{C.RESET}"
        f"  |  Reward: {C.BOLD}{reward:>+7.1f}{C.RESET}"
        f"  |  Total: {C.BOLD}{total_reward:>+8.1f}{C.RESET}"
    )
    if reason:
        reason_color = C.GREEN if reason == "reached_exit" else (
            C.RED if reason == "hit_fire" else C.YELLOW
        )
        lines.append(f"  Event: {reason_color}{C.BOLD}{reason}{C.RESET}")
    lines.append("")

    return "\n".join(lines)


def clear_screen() -> None:
    """Clear the terminal screen (cross-platform)."""
    os.system("cls" if os.name == "nt" else "clear")
