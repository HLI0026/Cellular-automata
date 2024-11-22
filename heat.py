import matplotlib.pyplot as plt
import numpy as np
import cv2



def update_grid(grid):
    padded_grid = np.pad(grid, pad_width=1)
    updated_grid = grid.copy()

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            neighborhood = padded_grid[i:i + 3, j:j + 3]
            updated_grid[i, j] = np.average(neighborhood)

    return updated_grid


def visualize_game_opencv(grid_size=(100, 100), steps = 100, cell_size=5,  random_seed=None):
    grid =np.zeros(grid_size)
    # grid[70,70] =200 # simulace vybuchu bomby
    window_name = "Heat"
    cv2.namedWindow(window_name)


    average_temp = []
    for s in range(steps):

        # grid[70, 70] = 50000  #
        # grid[130, 130] = 1000
        #
        # grid[70,70] = 100 * np.sin(s*np.pi/4)
        # grid[70, 70] = 100*np.exp(-s/60) * np.sin(s * np.pi / 12) #
        display_grid = cv2.resize(grid.astype(np.float64),
                                  (grid_size[1] * cell_size, grid_size[0] * cell_size),
                                  interpolation=cv2.INTER_NEAREST)

        cv2.imshow(window_name, display_grid)
        grid = update_grid(grid)
        average_temp.append(np.average(grid))
        if cv2.waitKey(50) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

    return average_temp
at = visualize_game_opencv(grid_size=(150, 150), steps = 500,cell_size=5, random_seed=69)
plt.plot(at)
plt.show()