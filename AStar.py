import numpy as np
import random
import matplotlib.pyplot as plt
from binaryHeap import BinaryHeap

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def find_unblocked_cell(grid):
    attempts = 0
    while True:
        x, y = random.randint(0, grid.shape[0] - 1), random.randint(0, grid.shape[1] - 1)
        if grid[x][y] == 1:
            print(f"Unblocked cell found at ({x}, {y}) after {attempts} attempts")
            return (x, y)
        attempts += 1
        if attempts > 10000:
            raise Exception("Unable to find an unblocked cell, check grid initialization.")

def a_star_search(grid, start, goal, BinaryHeap):
    open_set = BinaryHeap()
    open_set.push((0 + heuristic(start, goal), start))
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}

    while open_set:
        current_score, current = open_set.pop()
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1] and grid[neighbor[0]][neighbor[1]] == 1:
                tentative_g_score = gscore[current] + 1
                if tentative_g_score < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    open_set.push((fscore[neighbor], neighbor))

    return None  # Return None if no path is found

def visualize_path(grid, path, start, goal):
    plt.imshow(grid, cmap='gray')
    for x, y in path:
        plt.plot(y, x, 'ro')
    plt.plot(start[1], start[0], 'go')
    plt.plot(goal[1], goal[0], 'bo')
    plt.grid(True)
    plt.show()

def main():
    grid_path = 'generated_grids/grid_0.npy'
    grid = np.load(grid_path)
    start = find_unblocked_cell(grid)
    goal = find_unblocked_cell(grid)
    print("Start:", start, "Goal:", goal)
    path = a_star_search(grid, start, goal, BinaryHeap)
    if path:
        visualize_path(grid, path, start, goal)
    else:
        print("No path found")

if __name__ == "__main__":
    main()
