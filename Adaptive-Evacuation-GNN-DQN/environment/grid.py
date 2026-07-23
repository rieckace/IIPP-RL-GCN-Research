"""
grid.py

Defines the 2D grid used by the evacuation environment.
This module is independent of RL and only manages the grid.
"""

from typing import List
from environment.constants import CellType

class Grid:
    """Represents the building grid."""

    def __init__(self, rows: int = 10, cols: int = 10):
        self.rows = rows
        self.cols = cols
        self.grid = self.create_grid()

    def create_grid(self) -> List[List[int]]:
        """Create an empty grid."""
        return [
            [CellType.EMPTY for _ in range(self.cols)]
            for _ in range(self.rows)
        ]

    def reset(self) -> None:
        """Reset the grid."""
        self.grid = self.create_grid()

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
        """Clear a grid cell."""
        if self.is_valid_position(row, col):
            self.grid[row][col] = CellType.EMPTY

    def display(self) -> None:
        """Print grid in terminal."""

        symbols = {
            CellType.EMPTY: ".",
            CellType.WALL: "#",
            CellType.AGENT: "A",
            CellType.EXIT: "E",
            CellType.OBSTACLE: "O",
            CellType.SMOKE: "S",
            CellType.FIRE: "F",
            CellType.SENSOR: "I",
        }

        for row in self.grid:
            print(" ".join(symbols[cell] for cell in row))