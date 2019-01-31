import ImageUtils.plotter as my_plt
from ImageUtils import MultiplePlot
from ImageUtils.canny import *
from ImageUtils.utils import to_ndarray, test_intersection
from ImageUtils.hough_transform import hough_line, extract_hough_lines
import matplotlib.pyplot as plt
import sys
import cv2

rho_key = 'rho'
theta_key = 'theta'

original = to_ndarray('sudoku.jpg')
original1 = cv2.imread('sudoku.jpg')

drawer = MultiplePlot([10, 10], [1, 2])

img1 = gs_filter(original, 3)

img2, D = gradient_intensity(img1)


img3 = suppression(np.copy(img2), D)


img4, weak = threshold(np.copy(img3), 25, 255)


img5 = tracking(np.copy(img4), weak, 50)

img5[:, 400:] = 0

# plt.vlines(indexes, 0, original.shape[0], 'r')
drawer.add(original, "Original")


accumulator, thetas, rhos = hough_line(img5)
hough_lines = extract_hough_lines(accumulator, thetas, rhos, 8, 5, 10)
out = np.zeros_like(original)
lines = []
_x1, _y1, _x2, _y2 = [], [], [], []
for line in hough_lines:
    _line = []
    if abs(np.rad2deg(line[theta_key])) == 90:
        m = 0
        b = int(abs(line[rho_key]))
    elif abs(np.rad2deg(line[theta_key])) == 0:
        m = sys.maxsize
        b = int(abs(line[rho_key]))
    else:
        m = -1 * (np.cos(line[theta_key]) / np.sin(line[theta_key]))
        b = line[rho_key] / np.sin(line[theta_key])

    if abs(m) < 1:
        for x in range(out.shape[1]):
            y = int(m * x + b)
            if 0 < y < out.shape[0]:
                out[y, x] += 1
                _line.append((x, y))
    elif m == sys.maxsize:
        for y in range(out.shape[0]):
            out[y, b] += 1
            _line.append((b, y))
    else:
        for y in range(out.shape[0]):
            x = int((y - b) // m)
            if 0 < x < out.shape[1]:
                out[y, x] += 1
                _line.append((x, y))
    lineThickness = 2
    cv2.line(original1, _line[0], _line[-1], (0, 0, 255), lineThickness)
    lines.append(_line)
horizontal_found = 0
for i in range(len(lines)):
    for j in range(len(lines)):
        if i != j:
            test_intersection(lines[i][0],lines[i][-1], lines[j][0],lines[j][-1])
# cv2.startWindowThread()
# cv2.namedWindow("a")
cv2.imshow("a", original1)
cv2.waitKey(0)
# drawer.add(out, "Lines")
# drawer.show()

