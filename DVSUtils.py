import numpy as np


def delete_noise_by_neighbors(img, neighbor=[-1, 0, 1], min_neighbors_amount=3):
    height, width = img.shape
    out_img = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if img[i, j] == 0:
                continue
            counter = 0
            for k in neighbor:
                for l in neighbor:
                    try:
                        if img[i + k, j + l] == 255:
                            counter += 1
                    except:
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


def set_triangle_scope(img, mid_point=(660, 230), base_height=400):
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            if get_y_on_right_line(j, mid_point, base_height) > i or get_y_on_left_line(j, mid_point,
                                                                                        base_height) > i:
                img[i, j] = 0
    return img


def get_y_on_right_line(x, mid_point, base_height):
    x_middle_point, y_middle_point = mid_point
    m = base_height / x_middle_point
    b = y_middle_point - m * x_middle_point
    return m * x + b


def get_y_on_left_line(x, middle_point, base_height):
    x_middle_point, y_middle_point = middle_point
    m = - base_height / x_middle_point
    b = y_middle_point - m * x_middle_point
    return m * x + b
