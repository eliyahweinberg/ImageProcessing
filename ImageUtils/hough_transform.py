import numpy as np
from ImageUtils.utils import zero_frame

rho_key = 'rho'
theta_key = 'theta'


def hough_line(img):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = int( np.ceil(np.sqrt(width * width + height * height)) ) # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)
    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas))
    y_idxs, x_idxs = np.nonzero(img) # (row, col) indexes to edges
    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len)
            accumulator[rho, t_idx] += 1
    return accumulator, thetas, rhos


def extract_hough_lines(_accumulator, thetas, rhos, lines_num, clean_size, distance_constraint=1, avg_constraint=1):
    accumulator = np.copy(_accumulator)
    averaged = []
    lines = []
    j = 0
    while j < lines_num:
        idx = np.argmax(accumulator)
        rho = rhos[idx // accumulator.shape[1]]
        theta = thetas[idx % accumulator.shape[1]]
        for i in range(len(lines)):
            if abs(lines[i][rho_key] - rho) < distance_constraint and abs(lines[i][theta_key] - theta) < 5:
                if abs(lines[i][rho_key] - rho) < avg_constraint and lines[i] not in averaged:
                    lines[i][rho_key] = (lines[i][rho_key] + rho) // 2
                    averaged.append(lines[i])
                    break
                break
        else:
            lines.append({rho_key: rho, theta_key: theta})
            j += 1
        zero_frame(accumulator, clean_size, (idx // accumulator.shape[1], idx % accumulator.shape[1]))
    return lines



