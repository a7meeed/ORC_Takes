import numpy as np
import matplotlib.pyplot as plt
from heapq import heappop, heappush

# A* algorithm implementation
def astar(grid, start, goal):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: np.linalg.norm(np.array(start) - np.array(goal))}
    open_set = []
    heappush(open_set, (fscore[start], start))
    
    while open_set:
        _, current = heappop(open_set)
        
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data
        
        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j            
            tentative_g_score = gscore[current] + np.linalg.norm(np.array(current) - np.array(neighbor))
            if 0 <= neighbor[0] < grid.shape[0]:
                if 0 <= neighbor[1] < grid.shape[1]:                
                    if grid[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    continue
            else:
                continue
            
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
            
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in open_set]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + np.linalg.norm(np.array(neighbor) - np.array(goal))
                heappush(open_set, (fscore[neighbor], neighbor))
                
    return False

# Grid setup (0: open, 1: blocked)
grid = np.array([
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
])

start = (0, 0)
goal = (4, 4)
path = astar(grid, start, goal)
path = path + [start]
path = path[::-1]

# Plotting the grid and the path
plt.imshow(grid, cmap='Greys', origin='upper')
plt.plot([x[1] for x in path], [x[0] for x in path], marker = 'o')
plt.title("Path Between Two Points Using A*")
plt.show()
