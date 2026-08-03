"""
heuristics.py

Implements traditional search algorithms (e.g., A*) to provide 
heuristic guidance to the RL agent.
"""

import heapq
from typing import List, Tuple, Set
from environment.grid import Grid
from environment.constants import CellType

class AStarPlanner:
    """A* Pathfinding implementation for the evacuation grid.
    
    Treats walls as impassable, but ignores fire and smoke (the RL agent 
    must learn to deal with those).
    """

    @staticmethod
    def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    @classmethod
    def compute_path(cls, grid: Grid, start: Tuple[int, int], exits: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Compute the shortest path from start to the nearest exit.
        
        Args:
            grid: The current Grid object.
            start: The (row, col) position of the agent.
            exits: A set of (row, col) exit positions.
            
        Returns:
            A list of (row, col) coordinates representing the optimal path.
            If no path exists, returns an empty list.
        """
        if not exits:
            return []

        # Priority queue for A* (f_score, counter, current_node, path_so_far)
        # counter is used to break ties when f_scores are equal
        counter = 0
        queue = [(0, counter, start, [start])]
        
        # Track minimum g_score (cost from start) to reach each node
        g_scores = {start: 0}
        
        while queue:
            f, _, current, path = heapq.heappop(queue)
            
            if current in exits:
                return path
                
            for nr, nc in grid.get_neighbors(current[0], current[1]):
                # Treat walls, fire, and smoke as impassable so the dense reward
                # explicitly guides the RL agent along the safest path around them.
                cell_type = grid.get_cell(nr, nc)
                if cell_type in (CellType.WALL, CellType.FIRE, CellType.SMOKE):
                    continue
                    
                neighbor = (nr, nc)
                tentative_g = g_scores[current] + 1
                
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    
                    # Heuristic: min manhattan distance to ANY exit
                    h = min(cls.manhattan_distance(neighbor, exit_pos) for exit_pos in exits)
                    f_score = tentative_g + h
                    
                    counter += 1
                    heapq.heappush(queue, (f_score, counter, neighbor, path + [neighbor]))
                    
        return []
