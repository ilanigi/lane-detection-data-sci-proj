import numpy as np
import cv2
from cv2.cv2 import HoughLines
from matplotlib import pyplot as plt


def plot_by_points(img):
    img = delete_none_binary_pixels(img)
    img = delete_noise_by_neighbors(img, min_neighbors_amount=2)
    img = delete_noise_by_neighbors(img, min_neighbors_amount=1)
    points = get_data_from_image(img)
    y, x = zip(*points)
    plt.scatter(x, y, s=0.5, c='k')
    plt.show()


def hough_line_transform(img):
    img = delete_none_binary_pixels(img)
    img = delete_noise_by_neighbors(img, min_neighbors_amount=2)
    img = delete_noise_by_neighbors(img, min_neighbors_amount=1)
    # img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)

    img_unit8 = np.uint8(img)
    lines = HoughLines(img_unit8, 1, np.pi / 180, 20)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (255, 50, 20), 2)
        cv2.imshow('1', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


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
                    except IndexError:
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


def linear_equation(base_point, mid_point):
    base_x, base_y = base_point
    x_middle_point, y_middle_point, = mid_point
    m = (y_middle_point - base_y) / (x_middle_point - base_x)
    n = - (m * x_middle_point - y_middle_point)
    return lambda x: m * x + n


def set_triangle_scope(img, mid_point=(640, 200), base_height=650):
    height, width = img.shape
    left_linear_equation = linear_equation((0, base_height), mid_point)
    right_linear_equation = linear_equation((width, base_height), mid_point)
    for i in range(height):
        for j in range(width):
            if left_linear_equation(j) > i or right_linear_equation(j) > i:
                img[i, j] = 0
    return img


def delete_right_half(img):
    height, width = img.shape
    return img[:, :int(width / 2)]


def get_data_from_image(img):
    points = []
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            if img[i, j] == 0:
                continue
            else:
                points.append((height - i, j))

    return points
