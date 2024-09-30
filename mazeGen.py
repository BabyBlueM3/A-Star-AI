import numpy as np
import random
import os

class GridCreate:
    def __init__(self, rows=101, cols=101):
        self.rows = rows
        self.cols = cols
        self.grid = np.ones((rows, cols)) * -1  # Unknown cells (gray)
        self.visited = np.zeros((rows, cols), dtype=bool)  # Track visited cells
        self.generate_Grid()  # Start DFS from the top-left corner
    
    def generate_Grid(self):
        stack = [(0, 0)]
        self.visited[0, 0] = True

        while stack:
            row, col = stack.pop()

            # Decide whether unblocked (70% chance) or blocked (30% chance)
            if random.random() < 0.7:
                self.grid[row, col] = 1  # Unblocked (white)
            else:
                self.grid[row, col] = 0  # Blocked (black)

            # Get all possible neighbors and shuffle to ensure random paths
            neighbors = self.get_neighbors(row, col)
            random.shuffle(neighbors)
            
            # Univisted neighbors get added to stack
            for r, c in neighbors:
                if not self.visited[r, c]:
                    self.visited[r, c] = True
                    stack.append((r, c))

    def get_neighbors(self, row, col):
        neighbors = []
        if row > 0:  # Up
            neighbors.append((row - 1, col))
        if row < self.rows - 1:  # Down
            neighbors.append((row + 1, col))
        if col > 0:  # Left
            neighbors.append((row, col - 1))
        if col < self.cols - 1:  # Right
            neighbors.append((row, col + 1))
        return neighbors

def generate_and_save_grids(n=50, directory='generated_grids'):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(n):
        grid = GridCreate(101, 101)
        np.save(os.path.join(directory, f'grid_{i}.npy'), grid.grid)

# Usage
generate_and_save_grids()
