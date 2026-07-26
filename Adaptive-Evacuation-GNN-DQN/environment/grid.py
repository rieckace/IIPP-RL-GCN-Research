"""
grid.py

Defines the 2D grid used by the evacuation environment.
This module is independent of RL and only manages the grid.
"""

from typing import List, Tuple, Dict, Any
import numpy as np

from environment.constants import CellType


class Grid:
    """Represents the building grid."""

    def __init__(self, rows: int = 10, cols: int = 10):
        self.rows = rows
        self.cols = cols
        self.grid: List[List[int]] = self.create_grid()

    # ------------------------------------------------------------------
    # Creation & reset
    # ------------------------------------------------------------------
    def create_grid(self) -> List[List[int]]:
        """Create an empty grid filled with EMPTY cells."""
        return [
            [CellType.EMPTY for _ in range(self.cols)]
            for _ in range(self.rows)
        ]

    def reset(self) -> None:
        """Reset the grid to all-empty state."""
        self.grid = self.create_grid()

    # ------------------------------------------------------------------
    # Cell accessors
    # ------------------------------------------------------------------
    def is_valid_position(self, row: int, col: int) -> bool:
        """Check if a position lies inside the grid."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def get_cell(self, row: int, col: int) -> int:
        """Return value stored at a grid cell."""
        return self.grid[row][col]

    def set_cell(self, row: int, col: int, value: int) -> None:
        """Set a grid cell value."""
        if self.is_valid_position(row, col):
            self.grid[row][col] = value

    def clear_cell(self, row: int, col: int) -> None:
        """Clear a grid cell (set to EMPTY)."""
        if self.is_valid_position(row, col):
            self.grid[row][col] = CellType.EMPTY

    # ------------------------------------------------------------------
    # Neighbor queries (4-connected)
    # ------------------------------------------------------------------
    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Return valid 4-connected neighbor positions.

        Used for fire/smoke spread logic and later for building
        the adjacency list when converting the grid to a graph.
        """
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if self.is_valid_position(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    # ------------------------------------------------------------------
    # NumPy conversion
    # ------------------------------------------------------------------
    def to_numpy(self) -> np.ndarray:
        """Return a NumPy copy of the grid (shape: rows × cols, dtype int32).

        This is used to construct the observation for the Gymnasium env.
        """
        return np.array(self.grid, dtype=np.int32)

    # ------------------------------------------------------------------
    # Batch entity placement
    # ------------------------------------------------------------------
    def place_entities(self, entity_map: Dict[str, Any]) -> None:
        """Batch-place entities on the grid from a map config dict.

        Expected keys (all optional, values are lists of [row, col]):
            walls, exits, fire_sources, obstacles, sensors

        Args:
            entity_map: Dict mapping entity name → list of [row, col] positions.
        """
        key_to_cell = {
            "walls":        CellType.WALL,
            "exits":        CellType.EXIT,
            "fire_sources": CellType.FIRE,
            "obstacles":    CellType.OBSTACLE,
            "sensors":      CellType.SENSOR,
        }
        for key, cell_type in key_to_cell.items():
            positions = entity_map.get(key, [])
            for pos in positions:
                r, c = pos[0], pos[1]
                if self.is_valid_position(r, c):
                    self.grid[r][c] = cell_type

    # ------------------------------------------------------------------
    # Display (simple terminal printout)
    # ------------------------------------------------------------------
    def display(self) -> None:
        """Print grid in terminal using ASCII symbols."""
        symbols = {
            CellType.EMPTY:    ".",
            CellType.WALL:     "#",
            CellType.AGENT:    "A",
            CellType.EXIT:     "E",
            CellType.OBSTACLE: "O",
            CellType.SMOKE:    "S",
            CellType.FIRE:     "F",
            CellType.SENSOR:   "I",
        }
        for row in self.grid:
            print(" ".join(symbols.get(cell, "?") for cell in row))