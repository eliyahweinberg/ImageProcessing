from ImageUtils import MultiplePlot
from ImageUtils.kernels import *
import numpy as np
from ImageUtils.utils import convolve, zero_frame
import ImageUtils.plotter as my_plt
import matplotlib.pyplot as plt
import cv2

alpha = 0.06

original = np.zeros((200, 100))
original[30:170, 25:28] = original[30:170, 72:75] = 255
original[30:33, 28:72] = original[167:170, 28:72] = 255


drawer = MultiplePlot([10, 10], [1, 2])


Ix = convolve(original, kernel_sobel_x_3)
Iy = convolve(original, kernel_sobel_y_3)
IxIy = Ix * Iy
Ix_sq = Ix * Ix
Iy_sq = Iy * Iy
g_IxIy = convolve(IxIy, kernel_gaussian)
g_Ix_sq = convolve(Ix_sq, kernel_gaussian)
g_Iy_sq = convolve(Iy_sq, kernel_gaussian)

drawer.add(original, "original")

g_Ix_sq_g_Iy_sq = g_Ix_sq*g_Iy_sq
g_IxIy_g_IxIy = g_IxIy*g_IxIy

temp = g_Ix_sq_g_Iy_sq - g_IxIy_g_IxIy
har = g_Ix_sq_g_Iy_sq - g_IxIy_g_IxIy - alpha*(g_Ix_sq + g_Iy_sq)*(g_Ix_sq + g_Iy_sq)
x_s = []
y_s = []
for i in range(4):
    idx = np.argmax(har)
    y, x = idx // har.shape[1], idx % har.shape[1]
    x_s.append(x)
    y_s.append(y)
    zero_frame(har, 5, (y, x))

drawer.add(original, "Corners")
plt.plot(x_s, y_s, 'ro', markersize=8)


drawer.show()
