from ImageUtils import MultiplePlot
from ImageUtils.hough_transform import hough_line, extract_hough_lines
import numpy as np
import matplotlib.pyplot as plt

original = np.zeros((200, 100))
original[30:170, 25:26] = original[30:170, 74:75] = 255
original[30:31, 26:74] = original[169:170, 26:74] = 255

drawer = MultiplePlot([10, 10], [1, 2])
drawer.add(original, "Original")

accumulator, thetas, rhos = hough_line(original)
hough_lines = extract_hough_lines(accumulator, thetas, rhos, 4, 5)

out = np.zeros_like(original)

for i in hough_lines:
    i["rho"] = abs(int(i["rho"]))
    if np.rad2deg(i["theta"]) == -90:
        i["rho"] += 1
        for x_idx in range(out.shape[1]):
            out[i["rho"], x_idx] += 1
    elif np.rad2deg(i["theta"]) == 0:
        for y_idx in range(out.shape[0]):
            out[y_idx, i["rho"]] += 1

corners = np.where(out == 2)

# corners = [(a, b) for (a, b) in zip(corners_temp[0], corners_temp[1])]
y_idx = corners[0]
y_ends = y_idx + 1

drawer.add(original, "Corners")
plt.plot(corners[1], corners[0], 'ro')

drawer.show()
