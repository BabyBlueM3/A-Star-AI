import numpy as np
import random
import matplotlib.pyplot as plt
import os

class BinaryHeap:
    def __init__(self):
        self.heap = []

    def push(self, item):
        """Add item to the heap, maintaining the heap invariant."""
        self.heap.append(item)
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        """Remove and return the smallest item from the heap."""
        if len(self.heap) == 1:
            return self.heap.pop()
        root = self.heap[0]
        self.heap[0] = self.heap.pop()  # Move the last item to the root
        self._sift_down(0)
        return root

    def _sift_up(self, index):
        """Move the item at index up to its proper position in the heap."""
        item = self.heap[index]
        while index > 0:
            parent_index = (index - 1) >> 1
            parent = self.heap[parent_index]
            if item < parent:
                self.heap[index] = parent
                index = parent_index
                continue
            break
        self.heap[index] = item

    def _sift_down(self, index):
        """Move the item at index down to its proper position in the heap."""
        item = self.heap[index]
        end_index = len(self.heap)
        child_index = 2 * index + 1  # Left child index
        while child_index < end_index:
            right_child_index = child_index + 1
            if right_child_index < end_index and not self.heap[child_index] < self.heap[right_child_index]:
                child_index = right_child_index
            if self.heap[child_index] < item:
                self.heap[index] = self.heap[child_index]
                index = child_index
                child_index = 2 * index + 1
            else:
                break
        self.heap[index] = item

    def __len__(self):
        return len(self.heap)

    def __repr__(self):
        return f'{self.heap}'

def load_grid(path):
    return np.load(path)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def find_unblocked_cell(grid):
    while True:
        x = random.randint(0, grid.shape[0] - 1)
        y = random.randint(0, grid.shape[1] - 1)
        if grid[x][y] == 1:  # Assuming 1 represents unblocked cells
            return (x, y)

def a_star_search(grid, start, goal):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    open_heap = BinaryHeap()
    open_heap.push((fscore[start], start))
    
    while len(open_heap) > 0:
        current = open_heap.pop()[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1] and grid[neighbor[0]][neighbor[1]] == 1:
                tentative_g_score = gscore[current] + 1
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue
                if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in open_heap.heap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    open_heap.push((fscore[neighbor], neighbor))
    
    return False

def visualize_path(grid, path):
    plt.imshow(grid, cmap='binary')
    for (x, y) in path:
        plt.plot(y, x, 'ro')  # 'ro' to mark the path with red dots
    plt.plot(path[0][1], path[0][0], 'go')  # Start in green
    plt.plot(path[-1][1], path[-1][0], 'bo')  # Goal in blue
    plt.xlim(-0.5, grid.shape[1]-0.5)
    plt.ylim(-0.5, grid.shape[0]-0.5)
    plt.grid()
    plt.show()

def main():
    grid_path = 'generated_grids/grid_0.npy'  # Adjust path as necessary
    grid = load_grid(grid_path)
    start = find_unblocked_cell(grid)  # Find a random unblocked start position
    goal = find_unblocked_cell(grid)  # Find a random unblocked goal position

    print(f"Start: {start}, Goal: {goal}")

    path = a_star_search(grid, start, goal)
    if path:
        visualize_path(grid, path)
    else:
        print("No path found")

if __name__ == "__main__":
    main()
