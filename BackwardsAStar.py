import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from binaryHeap import BinaryHeap

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def find_unblocked_cell(grid):
    while True:
        x, y = random.randint(0, grid.shape[0] - 1), random.randint(0, grid.shape[1] - 1)
        if grid[x][y] == 1:
            return (x, y)

def repeated_a_star(grid, start, goal, binary_heap):
    global path  # Define path as global to make it accessible in animate
    path = []
    current = start
    open_set = binary_heap
    while current != goal:
        path_segment = a_star(grid, current, goal, open_set)
        if not path_segment:
            return None  # No path found
        for point in path_segment:
            if grid[point[0]][point[1]] == 0:  # Encounter unexpected obstacle
                break
            current = point
            path.append(point)
            if point == goal:
                return path
    return path

def a_star(grid, start, goal, open_set):
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    open_set.push((fscore[start], start))

    while open_set:
        _, current = open_set.pop()
        if current == goal:
            return reconstruct_path(came_from, current)

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1] and grid[neighbor[0]][neighbor[1]] == 1:
                tentative_g_score = gscore[current] + 1
                if tentative_g_score < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    open_set.push((fscore[neighbor], neighbor))
    return None

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def animate(i):
    global fig, ax
    if i < len(path):
        point = path[i]
        ax.plot(point[1], point[0], 'ro')  # Update plot with new point

def main():
    global fig, ax
    grid_path = 'generated_grids/grid_0.npy'
    grid = np.load(grid_path)
    start = find_unblocked_cell(grid)
    goal = find_unblocked_cell(grid)
    binary_heap = BinaryHeap()
    path = repeated_a_star(grid, start, goal, binary_heap)
    if path:
        fig, ax = plt.subplots()
        ax.imshow(grid, cmap='gray')
        ax.plot(start[1], start[0], 'go')  # Start in green
        ax.plot(goal[1], goal[0], 'bo')  # Goal in blue
        ani = FuncAnimation(fig, animate, frames=len(path), interval=200)
        plt.show()
    else:
        print("No path found")

if __name__ == "__main__":
    main()



# Heuristic function remains the same
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def find_unblocked_cell(grid):
    while True:
        x, y = random.randint(0, grid.shape[0] - 1), random.randint(0, grid.shape[1] - 1)
        if grid[x][y] == 1:
            return (x, y)

def repeated_backward_a_star(grid, start, goal, binary_heap):
    global path  # Define path as global to make it accessible in animate
    path = []
    current = goal  # Start from the goal
    open_set = binary_heap
    while current != start:  # Search until we reach the start
        path_segment = a_star(grid, current, start, open_set)  # Search towards the start
        if not path_segment:
            return None  # No path found
        for point in path_segment:
            if grid[point[0]][point[1]] == 0:  # Encounter unexpected obstacle
                break
            current = point
            path.append(point)
            if point == start:  # When we reach the start, return the path
                return path
    return path

# A* function remains the same
def a_star(grid, start, goal, open_set):
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    open_set.push((fscore[start], start))

    while open_set:
        _, current = open_set.pop()
        if current == goal:
            return reconstruct_path(came_from, current)

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1] and grid[neighbor[0]][neighbor[1]] == 1:
                tentative_g_score = gscore[current] + 1
                if tentative_g_score < gscore.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    open_set.push((fscore[neighbor], neighbor))
    return None

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def animate(i):
    global fig, ax
    if i < len(path):
        point = path[i]
        ax.plot(point[1], point[0], 'ro')  # Update plot with new point

def main():
    global fig, ax
    grid_path = 'generated_grids/grid_0.npy'
    grid = np.load(grid_path)
    start = find_unblocked_cell(grid)
    goal = find_unblocked_cell(grid)
    binary_heap = BinaryHeap()
    path = repeated_backward_a_star(grid, start, goal, binary_heap)
    if path:
        fig, ax = plt.subplots()
        ax.imshow(grid, cmap='gray')
        ax.plot(start[1], start[0], 'go')  # Start in green
        ax.plot(goal[1], goal[0], 'bo')  # Goal in blue
        ani = FuncAnimation(fig, animate, frames=len(path), interval=200)
        plt.show()
    else:
        print("No path found")

if __name__ == "__main__":
    main()

