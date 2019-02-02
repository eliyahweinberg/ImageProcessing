from ImageUtils.canny import *
from ImageUtils.utils import to_ndarray, intersect_lines
from ImageUtils.hough_transform import hough_line, extract_hough_lines
import sys
import cv2  # only for displaying

RHO_KEY = 'rho'
THETA_KEY = 'theta'
Y_AX = 1
X_AX = 0
lineThickness = 2

original = to_ndarray('sudoku.jpg')
img_show = cv2.imread('sudoku.jpg')

img1 = gs_filter(original, 3)

img2, D = gradient_intensity(img1)


img3 = suppression(np.copy(img2), D)


img4, weak = threshold(np.copy(img3), 25, 255)


img5 = tracking(np.copy(img4), weak, 50)
img5[:, 400:] = 0
accumulator, thetas, rhos = hough_line(img5)
hough_lines = extract_hough_lines(accumulator, thetas, rhos, 8, 5, 30, 10)
temp_lines = []
for line in hough_lines:
    _line = []
    if abs(np.rad2deg(line[THETA_KEY])) == 90:
        m = 0
        b = int(abs(line[RHO_KEY]))
    elif abs(np.rad2deg(line[THETA_KEY])) == 0:
        m = sys.maxsize
        b = int(abs(line[RHO_KEY]))
    else:
        m = -1 * (np.cos(line[THETA_KEY]) / np.sin(line[THETA_KEY]))
        b = line[RHO_KEY] / np.sin(line[THETA_KEY])

    if abs(m) < 1:
        for x in range(original.shape[1]):
            y = int(m * x + b)
            if 0 < y < original.shape[0]:
                _line.append((x, y))
    elif m == sys.maxsize:
        for y in range(original.shape[0]):
            _line.append((b, y))
    else:
        for y in range(original.shape[0]):
            x = int((y - b) // m)
            if 0 < x < original .shape[1]:
                _line.append((x, y))
    temp_lines.append(_line)

lines = [[line[0], line[-1]] for line in temp_lines]
for i in range(len(lines)):
    intrsct_points = []
    for line in lines:
        if lines[i] != line:
            x, y, z, r, s = intersect_lines(lines[i][0], lines[i][1], line[0], line[1])
            if r > 0 and 0 < x < original.shape[1] and 0 < y < original.shape[0]:
                intrsct_points.append((int(x), int(y)))
    lines[i].append(intrsct_points)

for line in lines:
    if abs(line[0][X_AX] - line[1][X_AX]) > abs(line[0][Y_AX] - line[1][Y_AX]):
        ax = X_AX
    else:
        ax = Y_AX
    min_point = line[2][0]
    max_point = line[2][0]
    for point in line[2]:
        if point[ax] > max_point[ax]:
            max_point = point
        elif point[ax] < min_point[ax]:
            min_point = point
    cv2.line(img_show, min_point, max_point, (0, 0, 255), lineThickness)


cv2.imshow("Original", img_show)
cv2.waitKey(0)


