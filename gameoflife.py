import matplotlib.pyplot as plt
import numpy as np
import cv2


def initialize_grid(size, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    return np.random.choice([0, 1], size=size)


def update_grid(grid):
    padded_grid = np.pad(grid, pad_width=1, mode='wrap')
    updated_grid = grid.copy()

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            neighborhood = padded_grid[i:i + 3, j:j + 3]
            alive_neighbors = np.sum(neighborhood) - padded_grid[i + 1, j + 1]
            if grid[i, j] == 1:
                if alive_neighbors < 2 or alive_neighbors > 3:
                    updated_grid[i, j] = 0
            else:
                if alive_neighbors in [3]:
                    updated_grid[i, j] = 1

    return updated_grid


def visualize_game_opencv(grid_size=(100, 100), steps = 100, cell_size=5,  random_seed=None):
    grid = initialize_grid(grid_size, random_seed)

    window_name = "Game of life"
    cv2.namedWindow(window_name)


    cells_alive = []
    for _ in range(steps):
        display_grid = cv2.resize(grid.astype(np.uint8) * 255,
                                  (grid_size[1] * cell_size, grid_size[0] * cell_size),
                                  interpolation=cv2.INTER_NEAREST)

        cv2.imshow(window_name, display_grid)

        grid = update_grid(grid)
        alive = np.sum(grid)
        cells_alive.append(alive)
        print(f"Alive cells: {alive}")
        if cv2.waitKey(50) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

    return cells_alive
ca = visualize_game_opencv(grid_size=(90, 90), steps = 500,cell_size=5, random_seed=69)
plt.plot(ca)
plt.show()