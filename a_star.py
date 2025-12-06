import heapq
import numpy as np
import matplotlib.pyplot as plt

# Maze representation
# 0 = free path
# 1 = wall

maze = np.array([
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0]
])

start = (0, 0)
goal = (4, 5)

# Manhattan heuristic
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(maze, start, goal):
    rows, cols = maze.shape
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, None))
    
    closed_set = set()
    parents = {}

    while open_set:
        f, g, current, parent = heapq.heappop(open_set)

        if current in closed_set:
            continue

        parents[current] = parent
        closed_set.add(current)

        if current == goal:
            # path found
            path = []
            while current:
                path.append(current)
                current = parents[current]
            path.reverse()
            return path

        x, y = current
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

        for nx, ny in neighbors:
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0:
                if (nx, ny) not in closed_set:
                    cost = g + 1
                    heapq.heappush(open_set, (cost + heuristic((nx, ny), goal), cost, (nx, ny), current))

    return None

# Run search
path = a_star_search(maze, start, goal)

# Show result
if path:
    print("Shortest Path:", path)
    print("Path Length:", len(path))
else:
    print("No path found")

# Visualization
def visualize_path(maze, path):
    maze_display = maze.copy()
    for r, c in path:
        maze_display[r][c] = 2  # mark path

    plt.imshow(maze_display, cmap="gray")
    plt.title("A* Path Finder")

    px, py = zip(*path)
    plt.plot(py, px)
    plt.scatter([start[1], goal[1]], [start[0], goal[0]])

    plt.show()

if path:
    visualize_path(maze, path)