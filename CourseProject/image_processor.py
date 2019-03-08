import cv2
import numpy as np
import ImageUtils.plotter as my_plt
from ImageUtils import MultiplePlot


IMAGES = ['1', '20', '39', '58', '78']

kernel_d = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], np.uint8)

LEFT_THRES = 6
TOP_THRES = 35
RIGHT_THRES = 30
ANGLE_THRES = 0.01
VERTICAL_ANGLE = 3.1241393
INTERNAL_THRESH = 10
TOP = 0
BOTTOM = 1
LEFT = 2
RIGHT = 3
RHO = 0
THETA = 1


def right_drop(right_lines):
    if abs(abs(right_lines[-2][RHO]) - abs(right_lines[-1][RHO])) > LEFT_THRES:
        del right_lines[-2:]
        return
    del right_lines[-1:]


def left_drop(left_lines):
    if len(left_lines) > 1 and abs(abs(left_lines[0][RHO]) - abs(left_lines[1][RHO])) > RIGHT_THRES:
        del left_lines[0]


def correct_horizontal_angles(top_line, bottom_line):
    if abs(top_line[THETA] - bottom_line[THETA]) > ANGLE_THRES:
        top_line[THETA] -= 0.02


def correct_vertical_angles(_lines):
    corrected = [[_rho, _theta] for _rho, _theta in _lines if abs(_theta - VERTICAL_ANGLE) < ANGLE_THRES]
    if len(corrected) < 1:
        _rho = _lines[0][RHO]
        if _rho > 0:
            _rho -= 7
            _rho *= -1

        corrected = [[_rho, VERTICAL_ANGLE]]
    return corrected


for file in IMAGES:
    print("Image: {0}".format(file))
    drawer = MultiplePlot([20, 20], [2, 2])
    image = cv2.imread('ImageInputs/{0}.tif'.format(file))
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray,70,255,cv2.THRESH_BINARY)
    opened = thresh
    for i in range(35):
        opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN, np.ones((3, 3)))

    origin = opened
    invert = np.invert(origin)

    filler = np.zeros_like(origin)
    filler[0, 0] = 255

    last = np.zeros_like(filler)

    while not np.array_equal(last, filler):
        last = filler
        dilated = cv2.dilate(filler, kernel_d, iterations=1)
        filler = np.bitwise_and(dilated, invert)

    filler = np.invert(filler)
    filled = origin + filler

    t_blur = cv2.GaussianBlur(filled, (13, 13), 0)

    t_edges = cv2.Canny(t_blur, 100, 255, 7)

    lines = cv2.HoughLines(t_edges, 1, np.pi / 180, 120)
    sorted_lines = [[], [], [], []]
    temp_horizontal, temp_vertical = [], []
    y_size, x_size = np.shape(origin)

    for line in lines:
        rho, theta = line[0]
        if abs(theta) < 1 or abs(theta) > 2:
            temp_vertical.append([rho, theta])
        else:
            temp_horizontal.append([rho, theta])


        # print(line)
        # cv2.imshow("Original", image)
        # cv2.waitKey(0)
    for temp_line in temp_horizontal:
        if abs(temp_line[0]) < y_size / 2:
            sorted_lines[TOP].append(temp_line)
        else:
            sorted_lines[BOTTOM].append(temp_line)
    for temp_line in temp_vertical:
        if abs(temp_line[0]) < x_size / 2:
            sorted_lines[LEFT].append(temp_line)
        else:
            sorted_lines[RIGHT].append(temp_line)

    for i in range(len(sorted_lines)):
        sorted_lines[i].sort(key=lambda x: abs(x[0]))

    right_drop(sorted_lines[RIGHT])
    left_drop(sorted_lines[LEFT])
    sorted_lines[TOP] = sorted_lines[TOP][:1]
    correct_horizontal_angles(sorted_lines[TOP][0], sorted_lines[BOTTOM][0])
    sorted_lines[LEFT] = correct_vertical_angles(sorted_lines[LEFT])
    sorted_lines[RIGHT] = correct_vertical_angles(sorted_lines[RIGHT])

    print("Top:")
    print(sorted_lines[TOP])
    print('Bottom:')
    print(sorted_lines[BOTTOM])
    print('Left:')
    print(sorted_lines[LEFT])
    print('Right:')
    print(sorted_lines[RIGHT])

    if sorted_lines[TOP][0][RHO] > TOP_THRES:
        sorted_lines[TOP][0][RHO] = TOP_THRES
    sorted_lines[BOTTOM] = sorted_lines[BOTTOM][:1]

    final_lines = [sorted_lines[TOP][0], sorted_lines[BOTTOM][0], sorted_lines[RIGHT][0]]
    if len(sorted_lines[LEFT]) > 1 and abs(sorted_lines[LEFT][0][RHO] - sorted_lines[LEFT][1][RHO]) > INTERNAL_THRESH:
        final_lines.append(sorted_lines[LEFT][0])
    else:
        final_lines.append(sorted_lines[LEFT][-1])

    for rho, theta in final_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    drawer.add(filled, "")
    drawer.add(t_blur, "")
    drawer.add(t_edges, "")
    drawer.add(image, '')
    drawer.show()
    print('\n\n')


