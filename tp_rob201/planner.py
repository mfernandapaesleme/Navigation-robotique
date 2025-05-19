import numpy as np
import heapq
from scipy.ndimage import binary_dilation
from occupancy_grid import OccupancyGrid


class Planner:
    """Simple occupancy grid Planner using A* algorithm"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        
    def set_occupancy_grid(self, occupancy_grid):
        """Atualiza a grade de ocupação"""
        self.grid = occupancy_grid
        
    def occupancy_grid_threshold(self, occupancy_grid, threshold=25):
        """
        Aplica um threshold na grade de ocupação
        - Valores > threshold viram 1 (obstáculo)
        - Valores <= threshold viram 0 (livre)
        """
        thresholded_grid = np.where(occupancy_grid > threshold, 1, 0)
        return thresholded_grid
    
    def occupancy_grid_dilate(self, occupancy_grid, radius=1):
        """
        Dilata os obstáculos para criar uma margem de segurança
        Isso faz com que os obstáculos fiquem "maiores" no planejamento
        """
        structure = np.ones((2*radius+1, 2*radius+1))
        dilated_grid = binary_dilation(occupancy_grid, structure=structure)
        return dilated_grid.astype(int)
    
    def get_neighbours(self, current_cell):
        x, y = current_cell
        neighbours = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if (0 < x + dx < self.grid.x_max_map) and (0 < y + dy < self.grid.y_max_map):
                    if self.grid.occupancy_map[x + dx, y + dy] < 35:
                        neighbour = (x + dx, y + dy)
                        neighbours.append(neighbour)
                
        return neighbours
    
    def heuristic(self, cell1, cell2):
        """
        Heuristic function for A* search
        cell1 : (x, y) tuple, first cell
        cell2 : (x, y) tuple, second cell
        """
        return np.linalg.norm(np.array(cell1) - np.array(cell2))

    def plan(self, start, goal):
        """
        A* path planning algorithm to find a path from start to goal
        start : [x, y] nparray, starting position
        goal : [x, y] nparray, goal position
        """
        
        start = self.grid.conv_world_to_map(start[0], start[1])
        goal = self.grid.conv_world_to_map(goal[0], goal[1])

        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), start))

        # For node n , cameFrom [ n] is the node immediately preceding it on the cheapest path from the start to n currently known
        came_from = {}

        # For node n , gScore [n] is the cost of the cheapest path from start to n currently known
        g_score = {tuple(start): 0}

        #  For node n , fScore [n] := gScore [ n] + h( n). fScore [n ] represents our current best guess as to 
        #  how cheap a path could be from start to finish if it goes through n.
        f_score = {tuple(start): self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]
            if np.array_equal(current, goal):
                print("Path found reconstructing")
                return self.reconstruct_path(came_from, current)

            for neighbour in self.get_neighbours(current):
                tentative_g_score = g_score[tuple(current)] + self.heuristic(current, neighbour)

                if tuple(neighbour) not in g_score or tentative_g_score < g_score[tuple(neighbour)]:
                    came_from[tuple(neighbour)] = current
                    g_score[tuple(neighbour)] = tentative_g_score
                    f_score[tuple(neighbour)] = tentative_g_score + self.heuristic(neighbour, goal)
                    heapq.heappush(open_set, (f_score[tuple(neighbour)], neighbour))

        return None  # Se não encontrou caminho
    


    def reconstruct_path(self, came_from, current):
        path = [current]
        while tuple(current) in came_from:
            current = came_from[tuple(current)]
            path.append(current)
        path.reverse()
        return path


    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        # Will use TP1 wall follow instead
        return goal
