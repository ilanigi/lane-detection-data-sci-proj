import numpy as np
from cv2.cv2 import imread, cvtColor, COLOR_BGR2GRAY


def general(image_name, min_neighbors_amount_list = [2, 1]):
    img = imread(image_name)
    img = cvtColor(img, COLOR_BGR2GRAY)
    img = delete_none_binary_pixels(img)
    for min_neighbor in min_neighbors_amount_list:
        img = delete_noise_by_neighbors(img, min_neighbors_amount=min_neighbor)
    return img


def delete_noise_by_neighbors(img, kernel=[-1, 0, 1], min_neighbors_amount=3):
    height, width = img.shape
    out_img = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if img[i, j] == 0:
                continue
            counter = 0
            for k in kernel:
                for m in kernel:
                    try:
                        if img[i + k, j + m] == 255:
                            counter += 1
                    except IndexError:
                        # kernel out of picture bound
                        pass
            # -1 ==> don't count a pixel as its own neighbor!
            if counter - 1 > min_neighbors_amount:
                out_img[i, j] = 255

    return out_img


def delete_none_binary_pixels(img):
    threshold = 240
    height, width = img.shape

    for i in range(height):
        for j in range(width):
            if img[i, j] > threshold:
                img[i, j] = 255
            else:
                img[i, j] = 0

    return img


def image_to_data(img):
    points = []
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            if img[i, j] == 0:
                continue
            else:
                points.append((j, i))

    return points


def set_linear_equation(point_1, point_2):
    x_1, y_1 = point_1
    x_2, y_2 = point_2
    m = (y_2 - y_1)/(x_2 - x_1)
    n = y_1 - m*x_1
    return lambda x: m * x + n


def get_data_from_parallelogram(img, par):
    upper_left, bottom_right, length = par
    x_left, y_up_left = upper_left
    x_right, y_btm_right = bottom_right
    y_up_right = y_btm_right - length
    y_btm_left = y_up_left + length

    upper_line = set_linear_equation(upper_left, (x_right, y_up_right))
    btm_line = set_linear_equation(bottom_right, (x_left, y_btm_left))
    points = []

    height, width = img.shape

    for y in range(min(y_up_right, y_up_left), max(y_btm_right, y_btm_left)):
        for x in range(x_left, x_right):
            if img[y, x] == 0:
                continue
            else:
                if upper_line(x) <= y <= btm_line(x):
                    points.append((x, y))

    return points
