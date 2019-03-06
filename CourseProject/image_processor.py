import cv2
import numpy as np
import ImageUtils.plotter as my_plt
from ImageUtils import MultiplePlot


files = ['1', '20', '39', '58', '78']

kernel_d = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], np.uint8)


for file in files:
    print("Image: {0}".format(file))
    drawer = MultiplePlot([20, 20], [2, 2])
    image = cv2.imread('ImageInputs/{0}.tif'.format(file))

    # drawer.add(image, "Original")
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray,70,255,cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
    opened = thresh
    for i in range(35):
        # t_eroded = cv2.erode(opened, np.ones((3, 3)), iterations=1)
        # opened = cv2.dilate(t_eroded, np.ones((3, 3)), iterations=1)
        opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN, np.ones((3, 3)))

    # t_eroded = cv2.erode(dilated, np.ones((5, 5)), iterations=1)

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

    lines = cv2.HoughLines(t_edges, 1, np.pi / 180, 125)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        print(line)
    drawer.add(filled, "")
    drawer.add(t_blur, "")
    drawer.add(t_edges, "")
    drawer.add(image, '')
    drawer.show()
    print('\n\n')




# image[dst>0.01*dst.max()]=[0,0,255]
# cv2.imshow("lines", image)
# cv2.waitKey(0)
# drawer.add(image, "Original")
# gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#
# thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
# t_eroded = cv2.erode(thresh, np.ones((3, 3)), iterations=1)
# i_eroded = cv2.erode(gray, np.ones((3, 3)), iterations=1)
# # dilated = cv2.dilate(eroded, np.ones((3, 3)), iterations=1)
#
#
# t_blur = cv2.GaussianBlur(t_eroded, (5, 5), 0)
# i_blur = cv2.GaussianBlur(i_eroded, (5, 5), 0)
#
# t_edges = cv2.Canny(t_blur, 100, 255, 7)
# i_edges = cv2.Canny(i_blur, 100, 255, 7)
#
#
# edges = i_edges + t_edges
#
#
# lines = cv2.HoughLines(t_blur, 1, np.pi / 180, 350)
# for line in lines:
#     rho, theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#
#     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
#
#
# drawer.add(t_eroded, "")
# drawer.add(t_blur, "")
# drawer.show()