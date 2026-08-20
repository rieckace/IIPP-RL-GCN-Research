"""
building.py

Manages building-level dynamics: fire spread and smoke propagation.
Isolated from RL logic so dynamics can be tested independently.
"""

import random
from typing import Optional

from environment.constants import CellType
from environment.grid import Grid
from environment.state import EnvironmentState


class Building:
    """Controls environment dynamics that happen each timestep.

    Fire spreads probabilistically to adjacent empty/smoke cells.
    Smoke fills cells within a Manhattan radius around fire sources.
    """

    def __init__(
        self,
        fire_spread_probability: float = 0.01,
        smoke_radius: int = 1,
    ) -> None:
        """
        Args:
            fire_spread_probability: Probability per timestep that fire
                spreads from a burning cell to each adjacent valid cell.
            smoke_radius: Manhattan distance within which smoke appears
                around every fire cell.
        """
        self.fire_spread_prob = fire_spread_probability
        self.smoke_radius = smoke_radius

    # ------------------------------------------------------------------
    # Fire spread
    # ------------------------------------------------------------------
    def spread_fire(
        self,
        grid: Grid,
        state: EnvironmentState,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Spread fire to adjacent empty or smoke cells.

        For each current fire cell, each 4-connected neighbor that is
        EMPTY or SMOKE has a chance of catching fire. Fire never spreads
        into walls, exits, obstacles, or cells that are already on fire.

        Fire spreads indefinitely — there is no burn-out. This creates
        natural time-pressure for the agent to evacuate quickly.

        Args:
            grid:  The current Grid.
            state: The current EnvironmentState (fire_cells will be mutated).
            rng:   Optional seeded Random instance for reproducibility.
        """
        if rng is None:
            rng = random.Random()

        new_fires: set[tuple[int, int]] = set()

        for fr, fc in list(state.fire_cells):
            for nr, nc in grid.get_neighbors(fr, fc):
                if (nr, nc) in state.fire_cells:
                    continue  # already burning
                cell = grid.get_cell(nr, nc)
                if cell in (CellType.EMPTY, CellType.SMOKE):
                    if rng.random() < self.fire_spread_prob:
                        new_fires.add((nr, nc))

        # Apply new fires
        for r, c in new_fires:
            state.fire_cells.add((r, c))
            # Remove from smoke if it was smoke before
            state.smoke_cells.discard((r, c))

    # ------------------------------------------------------------------
    # Smoke propagation
    # ------------------------------------------------------------------
    def spread_smoke(
        self,
        grid: Grid,
        state: EnvironmentState,
    ) -> None:
        """Generate smoke around all fire cells.

        Smoke appears in every EMPTY cell within `smoke_radius`
        Manhattan distance of any fire cell. Smoke does not overwrite
        walls, exits, obstacles, fire, or agents.

        This is recalculated from scratch each step (not cumulative),
        ensuring smoke always reflects the current fire state.

        Args:
            grid:  The current Grid.
            state: The current EnvironmentState (smoke_cells will be set).
        """
        smoke: set[tuple[int, int]] = set()

        for fr, fc in state.fire_cells:
            for dr in range(-self.smoke_radius, self.smoke_radius + 1):
                for dc in range(-self.smoke_radius, self.smoke_radius + 1):
                    if abs(dr) + abs(dc) > self.smoke_radius:
                        continue  # outside Manhattan radius
                    if dr == 0 and dc == 0:
                        continue  # fire cell itself
                    nr, nc = fr + dr, fc + dc
                    if not grid.is_valid_position(nr, nc):
                        continue
                    if (nr, nc) in state.fire_cells:
                        continue  # don't put smoke on fire
                    cell = grid.get_cell(nr, nc)
                    if cell in (CellType.EMPTY, CellType.AGENT):
                        smoke.add((nr, nc))

        state.smoke_cells = smoke

    # ------------------------------------------------------------------
    # Combined dynamics step
    # ------------------------------------------------------------------
    def step(
        self,
        grid: Grid,
        state: EnvironmentState,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Execute one timestep of environment dynamics.

        Order: spread fire → recalculate smoke → sync to grid.

        Args:
            grid:  The current Grid.
            state: The current EnvironmentState.
            rng:   Optional seeded Random for fire spread reproducibility.
        """
        self.spread_fire(grid, state, rng)
        self.spread_smoke(grid, state)
        state.sync_to_grid(grid)
