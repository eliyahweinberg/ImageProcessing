import cv2 as cv
import numpy as np
from ImageUtils import MultiplePlot

kernel_erosion = np.ones((3,3))
kernel_dilation = np.ones((5,5))
drawer = MultiplePlot([10, 10], [1, 2])


original = cv.imread('sudoku.png', 0)

drawer.add(original, "Original")


erosion = cv.erode(original, kernel_erosion, iterations=1)
dilation = cv.dilate(original, kernel_dilation, iterations=1)

drawer.add(dilation, "After Dilation")
drawer.show()
