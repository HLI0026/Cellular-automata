import numpy as np
import matplotlib.pyplot as plt
import cv2
def rule_general(initial_state, steps,rule):
    grid = np.zeros((steps, len(initial_state)+2), dtype=np.int16)
    grid[0,1:-1] = initial_state

    for t in range(1, steps):
        for i in range(1, len(initial_state) - 1):
            if rule & 1 << ((grid[t - 1, i - 2] << 4)+ (grid[t - 1, i -1 ] << 3) + (grid[t - 1, i] << 2)+ (grid[t - 1, i +1] << 1)+ (grid[t - 1, i + 2])):
                grid[t, i] = 1
            else:
                grid[t, i] = 0


    return grid

def visualize(grid):
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap="binary", interpolation="nearest")
    plt.show()

size = 301
steps = 200
# initial_state = np.zeros(size, dtype=np.int8)
# initial_state[size // 2] = 1
initial_state = np.random.choice([0, 1], size=size)
grid = rule_general(initial_state, steps, rule=980)
visualize(grid)
# rules to try
# 870 s random
# 772
# 950 s random
# 980 s random -