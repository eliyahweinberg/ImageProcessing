import cv2 as cv
import numpy as np
from ImageUtils import MultiplePlot

drawer = MultiplePlot([10, 10], [1, 2])

kernel_d = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], np.uint8)


origin = cv.imread("rise_hollow.jpg", 0)
origin[origin > 150] = 255
origin[origin < 150] = 0
invert = np.invert(origin)


filler = np.zeros_like(origin)
filler[0, 0] = 255

last = np.zeros_like(filler)

while not np.array_equal(last, filler):
    last = filler
    dilated = cv.dilate(filler, kernel_d, iterations=1)
    filler = np.bitwise_and(dilated, invert)

filler = np.invert(filler)
filled = origin + filler
drawer.add(origin, "Origin")
drawer.add(filled, "Filled")

drawer.show()
