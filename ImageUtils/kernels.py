import numpy as np

kernel_sobel = np.array([
    [-2, -1, 0],
    [-1, 0, 1],
    [0, 1, 2]
])

kernel_average = (1/9) * np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])

kernel_noise_remove = np.array([
    [2, 2, 2],
    [2, -1, 2],
    [2, 2, 2]
])

kernel_laplacian_5 = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

kernel_laplacian_8 = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

kernel_gaussian = (1/256) * np.array([[1, 4, 6, 4, 1],
                               [4, 16, 24, 16, 4],
                               [6, 24, 36, 24, 6],
                               [4, 16, 24, 16, 4],
                               [1, 4, 6, 4, 1]])

kernel_dx = np.array([[-1,0,1],
                      [-2,0,2],
                      [-1,0,1]
])

kernel_dy = np.array([[-1,-2,-1],
                      [0,0,0],
                      [1,2,1]
])