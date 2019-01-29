from ImageUtils import MultiplePlot
from ImageUtils.kernels import *
from ImageUtils.utils import convolve
from ImageUtils.utils import uniform_noise


original = np.zeros((200, 200))
original[50:150, 50:150] = 255

drawer = MultiplePlot([20, 10], [1, 3])
drawer.add(original, "Original")

noised = uniform_noise(original, 5)
drawer.add(noised, "Uniform noise")

cleaned = convolve(noised, kernel_noise_remove)
averaged = convolve(cleaned, kernel_average)


out = convolve(original, kernel_sobel)
out2 = abs(out)
drawer.add(out2, "After filter")
drawer.show()
