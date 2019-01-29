import cv2 as cv
import numpy as np
from ImageUtils import MultiplePlot
from ImageUtils.utils import threshold


drawer = MultiplePlot([10, 10], [1, 2])

original = cv.imread('rise.jpg', 0)

thresholded = threshold(original, 115)

opened = cv.morphologyEx(thresholded, cv.MORPH_OPEN, np.ones((7,7)))
erosion = cv.erode(opened, np.ones((5,5)), iterations=1)
res = opened - erosion

drawer.add(original, "Boundary")

drawer.add(res, "After")

drawer.show()
