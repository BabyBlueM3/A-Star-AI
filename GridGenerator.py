import numpy as np
import random
import matplotlib.pyplot as plt
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

    def visualize_and_save(self, directory, index):
        plt.figure(figsize=(10, 10))
        cmap = plt.cm.gray
        norm = plt.Normalize(-1, 1)
        plt.imshow(self.grid, cmap=cmap, norm=norm)
        plt.colorbar(ticks=[-1, 0, 1], label="Cell Type")
        plt.title(f"GridCreate {self.rows} x {self.cols} - Grid {index}")
        plt.axis('off')
        image_path = os.path.join(directory, f'grid_{index}.png')
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()

def generate_and_save_grids(n=50, directory='generated_grids'):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(n):
        grid = GridCreate(101, 101)
        np.save(os.path.join(directory, f'grid_{i}.npy'), grid.grid)
        grid.visualize_and_save(directory, i)

# Usage
generate_and_save_grids()
