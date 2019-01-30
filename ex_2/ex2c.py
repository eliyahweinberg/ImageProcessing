import ImageUtils.plotter as my_plt
from ImageUtils import MultiplePlot
from ImageUtils.canny import *
from ImageUtils.utils import to_ndarray
from ImageUtils.hough_transform import hough_line, extract_hough_lines
import matplotlib.pyplot as plt


original = to_ndarray('sudoku.jpg')

drawer = MultiplePlot([10, 10], [1, 2])
drawer.add(original, "Original")

img1 = gs_filter(original, 3)

img2, D = gradient_intensity(img1)


img3 = suppression(np.copy(img2), D)


img4, weak = threshold(np.copy(img3), 25, 255)


img5 = tracking(np.copy(img4), weak, 50)

# plt.vlines(indexes, 0, original.shape[0], 'r')


accumulator, thetas, rhos = hough_line(img5)
hough_lines = extract_hough_lines(accumulator, thetas, rhos, 8, 5, 10)
out = np.zeros_like(original)
for i in hough_lines:
    print("rho={0:.2f}, theta={1:.0f}".format(i['rho'], np.rad2deg(i['theta'])))
    i["rho"] = abs(int(i["rho"]))
    if -95 < np.rad2deg(i["theta"]) < -80:
        i["rho"] += 1
        for x_idx in range(out.shape[1]):
            out[i["rho"], x_idx] += 1
    elif -5 < np.rad2deg(i["theta"]) < 6:
        for y_idx in range(out.shape[0]):
            out[y_idx, i["rho"]] += 1

drawer.add(out, "Corners")


drawer.show()

