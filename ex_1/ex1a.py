from ImageUtils import MultiplePlot
from ImageUtils.kernels import *
from ImageUtils.utils import convolve

original = np.zeros((200, 200))
original[50:150, 50:150] = 255


drawer = MultiplePlot([10, 10], [1, 2])
drawer.add(original, "Original")
out = convolve(original, kernel_sobel)
out2 = abs(out)
drawer.add(out2, "After filter")
drawer.show()
