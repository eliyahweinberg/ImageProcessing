from ImageUtils import MultiplePlot
import cv2
from ImageUtils.kernels import *


original = cv2.imread('Lenna.png', 0)
if original is None:
    print("failed to read Lenna :(")
    exit

drawer = MultiplePlot([20, 10], [2, 2])

drawer.add(original, "Lenna")

img_blur = cv2.GaussianBlur(original, (5, 5), 0)
drawer.add(img_blur, "Bluring")

img_sharp = cv2.filter2D(original, -1, kernel_laplacian_5)
drawer.add(img_sharp, "Sharpening")

difference = original - img_sharp
drawer.add(difference, "Diff")

drawer.show()
