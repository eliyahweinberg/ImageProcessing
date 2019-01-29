from ImageUtils.kernels import *
from ImageUtils.utils import *
from ImageUtils.plotter import show
import matplotlib.pyplot as plt

original = to_ndarray('sudoku.jpg')
tresholded = threshold(original, 50)

averaged = convolve(tresholded, kernel_average)
derivated_x = convolve(averaged, kernel_dx)
positives = negative_to_zero(derivated_x)
cleaned = clean_frame(positives, 2)

indexes = max_with_frame(cleaned, 4, 10)
plt.vlines(indexes, 0, original.shape[0], 'r')
show(original)
