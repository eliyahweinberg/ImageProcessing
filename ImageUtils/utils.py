import imageio
import numpy as np
import math


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


def intersect_lines(pt1, pt2, ptA, ptB):
    """ this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)

        returns a tuple: (xi, yi, valid, r, s), where
        (xi, yi) is the intersection
        r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
        s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
            valid == 0 if there are 0 or inf. intersections (invalid)
            valid == 1 if it has a unique intersection ON the segment    """

    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1
    x2, y2 = pt2
    dx1 = x2 - x1
    dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA
    xB, yB = ptB
    dx = xB - x
    dy = yB - y

    # we need to find the (typically unique) values of r and s
    # that will satisfy
    #
    # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
    #
    # which is the same as
    #
    #    [ dx1  -dx ][ r ] = [ x-x1 ]
    #    [ dy1  -dy ][ s ] = [ y-y1 ]
    #
    # whose solution is
    #
    #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
    #    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
    #
    # where DET = (-dx1 * dy + dy1 * dx)
    #
    # if DET is too small, they're parallel
    #
    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE:
        return 0, 0, 0, 0, 0

    # now, the determinant should be OK
    DETinv = 1.0 / DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy * (x - x1) + dx * (y - y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

    # return the average of the two descriptions
    xi = (x1 + r * dx1 + x + s * dx) / 2.0
    yi = (y1 + r * dy1 + y + s * dy) / 2.0
    return xi, yi, 1, r, s


def test_intersection(pt1, pt2, ptA, ptB):
    print("Line segment #1 runs from", pt1, "to", pt2)
    print("Line segment #2 runs from", ptA, "to", ptB)

    result = intersect_lines(pt1, pt2, ptA, ptB)
    print("    Intersection result =", result)
