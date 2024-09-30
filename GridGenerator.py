import numpy as np
import random
import matplotlib.pyplot as plt

# Part 1: Setup

class GridCreate:
    def __init__(self, rows =101, cols= 101):
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

            #unblocked (70% chance) or blocked (30% chance)
            if random.random() < 0.7:
                self.grid[row, col] = 1  # Unblocked (white)
            else:
                self.grid[row, col] = 0  # Blocked (black)

            # Get all possible neighbors and shuffle to ensure random paths
            neighbors = self.get_neighbors(row, col)
            random.shuffle(neighbors)
            
            #Univisted neighbors get added to stack
            for r, c in neighbors:
                if not self.visited[r, c]:
                    self.visited[r, c] = True
                    stack.append((r,c))

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
    
    def visualize(self, save_as_image=False, image_path="gridcreated.png"):
        # Replace unknown (-1) with gray (0.5), unblocked (1) with white, and blocked (0) with black
        cmap = plt.cm.get_cmap('gray', 3)
        plt.imshow(self.grid, cmap=cmap, vmin=-1, vmax=1)
        plt.colorbar(ticks=[-1, 0, 1], label="Cell Type")
        plt.title(f"GridCreate {self.rows} x {self.cols}")
        
        if save_as_image:
            plt.savefig(image_path, dpi=300, bbox_inches='tight')  # Save as image with high DPI
            print(f"GridCreate saved as {image_path}")
        else:
            plt.show()


# Usage
grid_world = GridCreate(101, 101)
grid_world.visualize()
