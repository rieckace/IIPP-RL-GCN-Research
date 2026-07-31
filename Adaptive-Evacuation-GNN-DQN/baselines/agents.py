import numpy as np
import random
from collections import deque
import heapq

from environment.constants import CellType, Action

class RandomAgent:
    def act(self, obs, env=None):
        return random.randint(0, 3) # UP, DOWN, LEFT, RIGHT

class SearchAgent:
    """Base class for search-based algorithms"""
    def _parse_obs(self, obs, rows, cols):
        grid = obs.reshape(rows, cols)
        # Find agent and exit
        agent_pos = None
        exit_pos = None
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == CellType.AGENT:
                    agent_pos = (r, c)
                elif grid[r, c] == CellType.EXIT:
                    exit_pos = (r, c)
        return grid, agent_pos, exit_pos

    def _get_neighbors(self, r, c, grid, rows, cols):
        neighbors = []
        for dr, dc, act in [(-1, 0, Action.UP), (1, 0, Action.DOWN), (0, -1, Action.LEFT), (0, 1, Action.RIGHT)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                cell = grid[nr, nc]
                if cell not in (CellType.WALL, CellType.OBSTACLE, CellType.FIRE):
                    neighbors.append((nr, nc, act))
        return neighbors

    def _backtrack_path(self, parent_map, current):
        path = []
        while current in parent_map and parent_map[current][1] is not None:
            prev, act = parent_map[current]
            path.append(act)
            current = prev
        path.reverse()
        return path

class BFSAgent(SearchAgent):
    def act(self, obs, env=None):
        if env is None:
            raise ValueError("BFSAgent needs env to know rows/cols")
        grid, start, goal = self._parse_obs(obs, env.rows, env.cols)
        
        if not start or not goal:
            return Action.STAY.value
            
        queue = deque([start])
        parent = {start: (None, None)}
        
        while queue:
            curr = queue.popleft()
            if curr == goal:
                path = self._backtrack_path(parent, curr)
                return path[0].value if path else Action.STAY.value
                
            for nr, nc, act in self._get_neighbors(curr[0], curr[1], grid, env.rows, env.cols):
                if (nr, nc) not in parent:
                    parent[(nr, nc)] = (curr, act)
                    queue.append((nr, nc))
                    
        return Action.STAY.value # No path found

class DijkstraAgent(SearchAgent):
    def act(self, obs, env=None):
        # In a uniform cost grid, Dijkstra is identical to BFS. 
        # But we'll implement it with a priority queue for completeness.
        if env is None:
            raise ValueError("DijkstraAgent needs env to know rows/cols")
        grid, start, goal = self._parse_obs(obs, env.rows, env.cols)
        
        if not start or not goal:
            return Action.STAY.value
            
        pq = [(0, start)]
        distances = {start: 0}
        parent = {start: (None, None)}
        
        while pq:
            dist, curr = heapq.heappop(pq)
            if curr == goal:
                path = self._backtrack_path(parent, curr)
                return path[0].value if path else Action.STAY.value
                
            if dist > distances.get(curr, float('inf')):
                continue
                
            for nr, nc, act in self._get_neighbors(curr[0], curr[1], grid, env.rows, env.cols):
                new_dist = dist + 1 # Cost is always 1
                if new_dist < distances.get((nr, nc), float('inf')):
                    distances[(nr, nc)] = new_dist
                    parent[(nr, nc)] = (curr, act)
                    heapq.heappush(pq, (new_dist, (nr, nc)))
                    
        return Action.STAY.value

class AStarAgent(SearchAgent):
    def _heuristic(self, p1, p2):
        # Manhattan distance
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def act(self, obs, env=None):
        if env is None:
            raise ValueError("AStarAgent needs env to know rows/cols")
        grid, start, goal = self._parse_obs(obs, env.rows, env.cols)
        
        if not start or not goal:
            return Action.STAY.value
            
        pq = [(self._heuristic(start, goal), 0, start)] # (f_score, g_score, pos)
        g_scores = {start: 0}
        parent = {start: (None, None)}
        
        while pq:
            _, g, curr = heapq.heappop(pq)
            if curr == goal:
                path = self._backtrack_path(parent, curr)
                return path[0].value if path else Action.STAY.value
                
            if g > g_scores.get(curr, float('inf')):
                continue
                
            for nr, nc, act in self._get_neighbors(curr[0], curr[1], grid, env.rows, env.cols):
                new_g = g + 1
                if new_g < g_scores.get((nr, nc), float('inf')):
                    g_scores[(nr, nc)] = new_g
                    parent[(nr, nc)] = (curr, act)
                    f_score = new_g + self._heuristic((nr, nc), goal)
                    heapq.heappush(pq, (f_score, new_g, (nr, nc)))
                    
        return Action.STAY.value
