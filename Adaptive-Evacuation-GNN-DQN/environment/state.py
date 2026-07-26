"""
state.py

Manages the dynamic state of the evacuation environment:
agent positions, fire/smoke/exit cell tracking, and conversion
to observation vectors and graph representations.
"""

from typing import List, Tuple, Set, Dict, Any
import numpy as np

from environment.constants import CellType
from environment.grid import Grid


class EnvironmentState:
    """Tracks all mutable state that changes each timestep.

    Separating state from the grid lets us reason about entities
    independently and makes graph conversion straightforward.
    """

    def __init__(self) -> None:
        # Agent positions (single agent for Phase 1, list for future multi-agent)
        self.agent_positions: List[Tuple[int, int]] = []

        # Hazard tracking
        self.fire_cells: Set[Tuple[int, int]] = set()
        self.smoke_cells: Set[Tuple[int, int]] = set()

        # Structural elements
        self.exit_cells: Set[Tuple[int, int]] = set()
        self.wall_cells: Set[Tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def reset(
        self,
        agent_starts: List[List[int]],
        fire_sources: List[List[int]],
        exit_positions: List[List[int]],
        wall_positions: List[List[int]],
    ) -> None:
        """Re-initialise all state from a map config."""
        self.agent_positions = [(r, c) for r, c in agent_starts]
        self.fire_cells = {(r, c) for r, c in fire_sources}
        self.smoke_cells = set()
        self.exit_cells = {(r, c) for r, c in exit_positions}
        self.wall_cells = {(r, c) for r, c in wall_positions}

    # ------------------------------------------------------------------
    # Observation conversion
    # ------------------------------------------------------------------
    def to_observation(self, grid: Grid) -> np.ndarray:
        """Return a flat observation vector from the current grid.

        The observation is simply the flattened grid where each cell
        is encoded as its CellType integer value.

        Shape: (rows * cols,)  dtype: int32
        """
        return grid.to_numpy().flatten()

    # ------------------------------------------------------------------
    # Graph conversion (for future GNN consumption)
    # ------------------------------------------------------------------
    def to_graph(self, grid: Grid) -> Dict[str, np.ndarray]:
        """Convert the current grid state into a graph representation.

        Returns a dict with:
            node_features : np.ndarray, shape (num_nodes, num_cell_types)
                One-hot encoded cell types for each grid position.
                Node ordering is row-major: node_id = row * cols + col.

            edge_index : np.ndarray, shape (2, num_edges)
                COO-format edge list for 4-connected adjacency.
                Edges are bidirectional (i→j and j→i both present).

        This output is directly compatible with PyTorch Geometric's
        Data(x=node_features, edge_index=edge_index) constructor.
        """
        rows, cols = grid.rows, grid.cols
        num_nodes = rows * cols
        num_cell_types = len(CellType)

        # --- Node features: one-hot of cell type ---
        node_features = np.zeros((num_nodes, num_cell_types), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                node_id = r * cols + c
                cell_val = grid.get_cell(r, c)
                node_features[node_id, cell_val] = 1.0

        # --- Edge index: 4-connected adjacency (bidirectional) ---
        src_list: List[int] = []
        dst_list: List[int] = []
        for r in range(rows):
            for c in range(cols):
                node_id = r * cols + c
                for nr, nc in grid.get_neighbors(r, c):
                    neighbor_id = nr * cols + nc
                    src_list.append(node_id)
                    dst_list.append(neighbor_id)

        edge_index = np.array([src_list, dst_list], dtype=np.int64)

        return {
            "node_features": node_features,
            "edge_index": edge_index,
        }

    # ------------------------------------------------------------------
    # Sync state → grid
    # ------------------------------------------------------------------
    def sync_to_grid(self, grid: Grid) -> None:
        """Write the current entity state onto the grid.

        Call this after modifying agent/fire/smoke positions to keep
        the grid consistent for rendering and observation generation.
        """
        # Clear old dynamic entities (keep walls and exits)
        for r in range(grid.rows):
            for c in range(grid.cols):
                cell = grid.get_cell(r, c)
                if cell in (CellType.AGENT, CellType.FIRE, CellType.SMOKE):
                    grid.clear_cell(r, c)

        # Place fire first (so smoke doesn't overwrite fire cells)
        for r, c in self.fire_cells:
            grid.set_cell(r, c, CellType.FIRE)

        # Place smoke
        for r, c in self.smoke_cells:
            # Don't overwrite fire, walls, or exits with smoke
            if grid.get_cell(r, c) == CellType.EMPTY:
                grid.set_cell(r, c, CellType.SMOKE)

        # Place agents
        for r, c in self.agent_positions:
            grid.set_cell(r, c, CellType.AGENT)
