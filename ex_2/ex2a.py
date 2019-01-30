import ImageUtils.plotter as my_plt
from ImageUtils.canny import *
from ImageUtils.utils import to_ndarray
from ImageUtils.utils import max_with_frame
import matplotlib.pyplot as plt


original = to_ndarray('sudoku.jpg')


img1 = gs_filter(original, 3)

img2, D = gradient_intensity(img1)


img3 = suppression(np.copy(img2), D)


img4, weak = threshold(np.copy(img3), 25, 255)


img5 = tracking(np.copy(img4), weak, 50)
indexes = max_with_frame(img5, 4, 15)
plt.vlines(indexes, 0, original.shape[0], 'r')
my_plt.show(original)


