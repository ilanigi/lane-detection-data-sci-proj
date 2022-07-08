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


def set_linear_equation_by_y(point_1, point_2):
    x_1, y_1 = point_1
    x_2, y_2 = point_2
    m = (y_2 - y_1)/(x_2 - x_1)
    n = y_1 - m*x_1

    return lambda y: (y - n) / m


def get_data_from_parallelogram(img, par):
    upper_left, bottom_right, par_width = par
    x_up_left, y_up = upper_left
    x_btm_right, y_btm = bottom_right

    x_up_right = x_up_left + par_width
    x_btm_left = x_btm_right - par_width

    left_line = set_linear_equation(upper_left, (x_btm_left, y_btm))
    right_line = set_linear_equation(bottom_right, (x_up_right, y_up))
    points = []

    for y in range(y_up, y_btm):
        for x in range(min(x_btm_left,x_up_left),max(x_btm_right,x_up_right) ):
            if img[y, x] == 0:
                continue
            else:
                if left_line(x) <= y <= right_line(x):
                    points.append((x, y))

    return points
