import numpy as np
import random
import matplotlib.pyplot as plt
import os
from itertools import product  # Add this import statement
import time





class BinaryHeap:
    def __init__(self):
        self.heap = []

    def push(self, fscore, gscore, item):
        """Add item to the heap, maintaining the heap invariant."""
        # We push (fscore, -gscore, item) so that Python's min-heap will handle ties by preferring higher gscore
        self.heap.append((fscore, -gscore, item))
        self._sift_up(len(self.heap) - 1)

    def pop(self):
        """Remove and return the item with the smallest fscore and highest gscore in case of a tie."""
        if len(self.heap) == 1:
            return self.heap.pop()[2]  # Return the item (third element in tuple)
        root = self.heap[0][2]  # Store the item (third element in tuple)
        self.heap[0] = self.heap.pop()  # Move the last item to the root
        self._sift_down(0)
        return root

    def _sift_up(self, index):
        """Move the item at index up to its proper position in the heap."""
        item = self.heap[index]
        while index > 0:
            parent_index = (index - 1) >> 1
            parent = self.heap[parent_index]
            # Compare by fscore first, and if tied, by -gscore
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
            # Compare left and right child; choose the smaller fscore, then higher gscore in ties
            if right_child_index < end_index and not self.heap[child_index] < self.heap[right_child_index]:
                child_index = right_child_index
            # Compare with the current item
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

# Constants
GRID_SIZE = 101
NUM_ENVIRONMENTS = 50
BLOCK_PROBABILITY = 0.3
UNBLOCK_PROBABILITY = 0.7

# Directions for movement (up, down, left, right)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def generate_grid():
    # Initialize the grid where 1 = blocked, 0 = unblocked
    grid = np.ones((GRID_SIZE, GRID_SIZE), dtype=int)
    visited = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    # Start from a random cell
    stack = []
    start_row, start_col = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
    stack.append((start_row, start_col))
    visited[start_row, start_col] = True
    grid[start_row, start_col] = 0  # Mark as unblocked

    while stack:
        # Get the current position
        row, col = stack[-1]

        # Find all unvisited neighbors
        neighbors = []
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and not visited[r, c]:
                neighbors.append((r, c))

        if neighbors:
            # Randomly select a neighbor to visit
            next_row, next_col = random.choice(neighbors)

            # With 30% probability, mark as blocked, otherwise unblocked
            if random.random() < BLOCK_PROBABILITY:
                grid[next_row, next_col] = 1  # Block the cell
            else:
                grid[next_row, next_col] = 0  # Unblock the cell
                stack.append((next_row, next_col))

            # Mark the neighbor as visited
            visited[next_row, next_col] = True
        else:
            # Dead-end: backtrack
            stack.pop()

    return grid

def visualize_grid(grid):
    # Create a visual representation of the grid where 1 = blocked and 0 = unblocked
    plt.imshow(grid, cmap='binary', interpolation='none')
    plt.title('Gridworld')
    plt.show()
    
def save_grids(num_environments=NUM_ENVIRONMENTS):
    grids = []
    for i in range(num_environments):
        grid = generate_grid()
        grids.append(grid)
        '''
        # Save the grid to a .txt file
        filename = f'grid_{i}.txt'
        np.savetxt(filename, grid, fmt='%d')  # Save as integers (0s and 1s)
        print(f"Grid {i} saved as {filename}.")  # Optional: Print a message for confirmation
        '''
    return grids


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def find_unblocked_cell(grid):
    while True:
        x = random.randint(0, grid.shape[0] - 1)
        y = random.randint(0, grid.shape[1] - 1)
        if grid[x][y] == 0:  # Assuming 0 represents unblocked cells
            return (x, y)
        
def initialize_visibility_grid(grid, start, goal):
    visibility_grid = np.zeros_like(grid)
    visibility_grid[start[0], start[1]] = 1  # Reveal start position
    visibility_grid[goal[0], goal[1]] = 1  # Goal is always visible

    # Initially reveal adjacent cells around the start position
    reveal_adjacent_cells(start, grid, visibility_grid)

    # Also reveal the goal's adjacent cells
    reveal_adjacent_cells(goal, grid, visibility_grid)

    return visibility_grid



def reveal_adjacent_cells(pos, grid, visibility_grid):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
    visibility_grid[pos[0], pos[1]] = 1  # Reveal current position

    for direction in directions:
        neighbor = (pos[0] + direction[0], pos[1] + direction[1])
        if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
            visibility_grid[neighbor[0], neighbor[1]] = 1  # Mark neighbor as seen
            
            # Reveal surrounding cells for better pathfinding
            for adj_direction in directions:
                adj_neighbor = (neighbor[0] + adj_direction[0], neighbor[1] + adj_direction[1])
                if 0 <= adj_neighbor[0] < grid.shape[0] and 0 <= adj_neighbor[1] < grid.shape[1]:
                    visibility_grid[adj_neighbor[0], adj_neighbor[1]] = 1  # Reveal more around




def a_star_search_with_fog(grid, visibility_grid, start, goal):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}

    open_heap = BinaryHeap()

    # Push (fscore, gscore, node)
    open_heap.push(fscore[start], gscore[start], start)

    while len(open_heap) > 0:
        current = open_heap.pop()  # Corrected to get just the item

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Return the reversed path

        close_set.add(current)
        reveal_adjacent_cells(current, grid, visibility_grid)

        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if visibility_grid[neighbor[0]][neighbor[1]] == 1 and grid[neighbor[0]][neighbor[1]] == 0:
                    tentative_g_score = gscore[current] + 1
                    if tentative_g_score < gscore.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        gscore[neighbor] = tentative_g_score
                        fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        open_heap.push(fscore[neighbor], gscore[neighbor], neighbor)  # Corrected push

    print("No path found in Repeated A* search.")
    return None  # Returning None for consistency

def repeated_backward_a_star_with_fog(grid, start, goal):
    global path  # Define path as global to make it accessible in animate
    path = []
    current = goal  # Start from the goal
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    visibility_grid = initialize_visibility_grid(grid, start, goal)

    open_heap = BinaryHeap()
    open_heap.push(fscore[start], gscore[start], start)
    while current != start:  # Search until we reach the start
        path_segment = a_star_search_with_fog(grid, visibility_grid, current, start)  # Search towards the start
        if not path_segment:
            return None  # No path found
        for point in path_segment:
            if grid[point[0]][point[1]] == 1:  # Encounter unexpected obstacle
                break
            current = point
            path.append(point)
            visibility_grid[point[0]][point[1]] = 1  # Mark as seen

            # Reveal adjacent cells from the new position
            reveal_adjacent_cells(current, grid, visibility_grid)

            if current == start:  # When we reach the start, return the path
                return path
    return path


def repeated_a_star_with_fog(grid, start, goal):
    print(f"Start Position: {start}, Goal Position: {goal}")
    visibility_grid = initialize_visibility_grid(grid, start, goal)
    current_pos = start
    full_path = [start]
    '''
    #debugging output
    print("Initial Grid:")
    print(grid)
    '''
    '''
    #debugging output
    print(f"Start Position: {start}, Goal Position: {goal}")
    '''
    while current_pos != goal:


        # Perform A* search with fog of war (limited visibility)
        partial_path = a_star_search_with_fog(grid, visibility_grid, current_pos, goal)

        if not partial_path:
            print(f"No path found from {current_pos} to {goal}. Checking for alternatives.")
            # You could add logic here to try other paths or backtrack
            break  # For now, just break out if no path is found

        for step in partial_path:
            if grid[step[0], step[1]] == 1:  # Blocked cell
                visibility_grid[step[0], step[1]] = 1  # Mark as seen
                print(f"Blocked cell encountered at {step}. Re-planning from {current_pos}.")
                break
            else:
                visibility_grid[step[0], step[1]] = 1  # Mark as seen
                full_path.append(step)
                current_pos = step
                
                # Reveal adjacent cells from the new position
                reveal_adjacent_cells(current_pos, grid, visibility_grid)
        
                # Check if goal is reached
                if current_pos == goal:
                    print("Goal reached!")
                    return full_path

    print("No valid path found after multiple attempts.")
    return None  # Return None if no path found

# Global dictionary to store updated heuristic values
updated_heuristics = {}

def adaptive_heuristic(state, goal):
    # If state has an updated heuristic value, use it
    if state in updated_heuristics:
        return updated_heuristics[state]
    else:
        # Use the standard heuristic (e.g., Manhattan distance) if no updated value
        return heuristic(state, goal)

def adaptive_a_star_with_fog(grid, start, goal):
    print(f"Start Position: {start}, Goal Position: {goal}")
    visibility_grid = initialize_visibility_grid(grid, start, goal)
    current_pos = start
    full_path = [start]

    # First run of the adaptive A* search
    partial_path = adaptive_a_star_search_with_fog(grid, visibility_grid, current_pos, goal)

    if not partial_path:
        print(f"No path found from {current_pos} to {goal}.")
        return None  # Return None if no path found

    # Update heuristics based on the found path
    for i, step in enumerate(partial_path):
        # Update heuristic values based on g-values found during the search
        g_value = i  # Use the index as a g-value (could be improved)
        updated_heuristics[step] = g_value

    # Now run the adaptive A* search again with updated heuristics
    print("Re-running adaptive A* search with updated heuristics...")
    current_pos = start  # Reset current position
    full_path = [start]  # Reset full path

    while current_pos != goal:
        # Perform A* search with fog of war using updated heuristics
        partial_path = adaptive_a_star_search_with_fog(grid, visibility_grid, current_pos, goal)

        if not partial_path:
            print(f"No path found from {current_pos} to {goal}. Checking for alternatives.")
            break  # For now, just break out if no path is found

        for step in partial_path:
            if grid[step[0], step[1]] == 1:  # Blocked cell
                visibility_grid[step[0], step[1]] = 1  # Mark as seen
                print(f"Blocked cell encountered at {step}. Re-planning from {current_pos}.")
                break
            else:
                visibility_grid[step[0], step[1]] = 1  # Mark as seen
                full_path.append(step)
                current_pos = step
                
                # Reveal adjacent cells from the new position
                reveal_adjacent_cells(current_pos, grid, visibility_grid)

                # Check if goal is reached
                if current_pos == goal:
                    print("Goal reached!")
                    return full_path

    print("No valid path found after multiple attempts.")
    return None  # Return None if no path found

def adaptive_a_star_search_with_fog(grid, visibility_grid, start, goal):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: adaptive_heuristic(start, goal)}

    open_heap = BinaryHeap()

    # Push (fscore, gscore, node)
    open_heap.push(fscore[start], gscore[start], start)

    while len(open_heap) > 0:
        current = open_heap.pop()  # Corrected to get just the item

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Return the reversed path

        close_set.add(current)
        reveal_adjacent_cells(current, grid, visibility_grid)

        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if visibility_grid[neighbor[0]][neighbor[1]] == 1 and grid[neighbor[0]][neighbor[1]] == 0:
                    tentative_g_score = gscore[current] + 1
                    if tentative_g_score < gscore.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        gscore[neighbor] = tentative_g_score
                        fscore[neighbor] = tentative_g_score + adaptive_heuristic(neighbor, goal)
                        open_heap.push(fscore[neighbor], gscore[neighbor], neighbor)  # Corrected push

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
    grids = save_grids(num_environments=NUM_ENVIRONMENTS)
    grid = grids[0]
    start = find_unblocked_cell(grid)
    goal = find_unblocked_cell(grid)

    print(f"Grid dimensions: {grid.shape}, Start: {start}, Goal: {goal}")

    # Run Repeated A* with Fog
    start_time = time.time()
    path_repeated_a_star = repeated_a_star_with_fog(grid, start, goal)
    repeated_a_star_time = time.time() - start_time

    if path_repeated_a_star:
        visualize_path(grid, path_repeated_a_star)
        print(f"Repeated A* Path Length: {len(path_repeated_a_star)}")
        print(f"Repeated A* Execution Time: {repeated_a_star_time:.3f} seconds")
    else:
        print("No path found in Repeated A* with Fog.")
    # Run Repeated Backward A* with Fog
    start_time = time.time()
    path_repeated_backward_a_star = repeated_backward_a_star_with_fog(grid, start, goal)
    repeated_backward_a_star_time = time.time() - start_time

    if path_repeated_backward_a_star:
        visualize_path(grid, path_repeated_backward_a_star)
        print(f"Repeated Backward A* Path Length: {len(path_repeated_backward_a_star)}")
        print(f"Repeated Backward A* Execution Time: {repeated_backward_a_star_time:.3f} seconds")
    else:
        print("No path found in Repeated Backward A* with Fog.")
        
    # Run Adaptive A* with Fog
    start_time = time.time()
    path_adaptive_a_star = adaptive_a_star_with_fog(grid, start, goal)
    adaptive_a_star_time = time.time() - start_time

    if path_adaptive_a_star:
        visualize_path(grid, path_adaptive_a_star)
        print(f"Adaptive A* Path Length: {len(path_adaptive_a_star)}")
        print(f"Adaptive A* Execution Time: {adaptive_a_star_time:.3f} seconds")
    else:
        print("No path found in Adaptive A* with Fog.")



    # Compare path lengths if all paths are found
    if path_repeated_a_star and path_adaptive_a_star and path_repeated_backward_a_star:
       print(f"Path length comparison: Repeated A* ({len(path_repeated_a_star)}) vs Repeated Backward A* ({len(path_repeated_backward_a_star)}) vs Adaptive A* ({len(path_adaptive_a_star)})")



if __name__ == "__main__":
    main()
