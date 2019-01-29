import imageio
import numpy as np


def to_ndarray(img):
    im = imageio.imread(img, as_gray='True')
    im = im.astype('int32')
    img = np.asarray(im)
    return img


def round_angle(angle):
    """ Input angle must be in [0,180) """
    angle = np.rad2deg(angle) % 180
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        angle = 0
    elif 22.5 <= angle < 67.5:
        angle = 45
    elif 67.5 <= angle < 112.5:
        angle = 90
    elif 112.5 <= angle < 157.5:
        angle = 135
    return angle


def max_columns_with_frame(img, columns_num, frame_size):
    res = img.sum(axis=0)
    indexes = []
    x = 0
    while x < columns_num and len(indexes) < 5:
        max_val_index = np.argmax(res)
        if len(indexes) == 0:
            indexes.append(max_val_index)
            res[max_val_index] = 0
        else:
            for y in indexes:
                if abs(max_val_index - y) <= frame_size:
                    res[max_val_index] = 0
                    x -= 1
                    break
            else:
                indexes.append(max_val_index)
                res[max_val_index] = 0
        x += 1
    return indexes


def convolve(img, kernel):
    kernel_size, s = kernel.shape
    if kernel_size != s:
        raise ValueError("Kernel dimensions is not equals ")
    height, width = img.shape
    padding = kernel_size - 1
    copy_dimensions = int(padding/2)
    out = np.zeros_like(img)
    img_padded = np.zeros((height+padding, width+padding))
    img_padded[copy_dimensions:-copy_dimensions, copy_dimensions:-copy_dimensions] = img
    for x in range(height):
        for y in range(width):
            out[x, y] = (kernel * img_padded[x:x+kernel_size, y:y+kernel_size]).sum()
    return out


def uniform_noise(img, percent):
    out = np.copy(img)
    noise_m = np.random.randint(0, 100, size=out.shape)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if 0 < noise_m[x, y] <= percent:
                out[x, y] = 100
    return out


def convolve2D(image, kernel):
    m, n = kernel.shape
    if m == n:
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y, x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel)
    return new_image


def negative_to_zero(img):
    out = np.copy(img)
    out[out < 0] = 0
    return out


def threshold(img, threshold):
    out = np.copy(img)
    out[out <= threshold] = 0
    out[out > threshold] = 1
    return out


def clean_frame(img, width):
    out = np.copy(img)
    out[:, 0:width] = 0
    out[:, -width:] = 0
    return out


def max_with_frame(img, columns_num, frame_size):
    maxes = np.zeros(img.shape[1])
    # corrections = np.zeros_like(maxes)
    res = img.sum(axis=0)
    indexes = []
    max_indexes = []
    max_val_index = np.argmax(res)
    indexes.append(max_val_index)
    maxes[max_val_index] = res[max_val_index]
    res[max_val_index] = 0
    for i in range(img.shape[1]):
        max_val_index = np.argmax(res)
        for y in indexes:
            if abs(max_val_index - y) <= frame_size:
                maxes[y] += res[max_val_index]
                res[max_val_index] = 0
                break
        else:
            indexes.append(max_val_index)
            maxes[max_val_index] = res[max_val_index]
            res[max_val_index] = 0

    for i in range(columns_num):
        max_val_index = np.argmax(maxes)
        maxes[max_val_index] = 0
        max_indexes.append(max_val_index)
    return max_indexes



