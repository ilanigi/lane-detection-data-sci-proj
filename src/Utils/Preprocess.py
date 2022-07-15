import numpy as np
from cv2.cv2 import imread, cvtColor, COLOR_BGR2GRAY

from Utils.Types import Parallelogram, Point


def general(image_name:str, min_neighbors_amount_list = [2, 1])->np.ndarray:
    img = imread(image_name)
    img = cvtColor(img, COLOR_BGR2GRAY)
    img = delete_none_binary_pixels(img)
    for min_neighbor in min_neighbors_amount_list:
        img = delete_noise_by_neighbors(img, min_neighbors_amount=min_neighbor)
    return img


def delete_noise_by_neighbors(img:np.ndarray, kernel=[-1, 0, 1], min_neighbors_amount=3):
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


def delete_none_binary_pixels(img:np.ndarray):
    threshold = 240
    height, width = img.shape

    for i in range(height):
        for j in range(width):
            if img[i, j] > threshold:
                img[i, j] = 255
            else:
                img[i, j] = 0

    return img


def image_to_data(img:np.ndarray):
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

"""
Get n and m for linear equation out of two points on it's line.
"""
def get_params_for_linear_equation(point_1:Point, point_2:Point)->Point:
    x_1, y_1 = point_1
    x_2, y_2 = point_2

    m = (y_2 - y_1)/(x_2 - x_1)   
    n = y_1 - m*x_1
    return  m , n


def get_data_from_parallelogram(img:np.ndarray, par:Parallelogram):
    height, width = img.shape

    upper_left, bottom_right, par_width = par
    x_up_left, y_up = upper_left
    x_btm_right, y_btm = bottom_right

    x_up_right = x_up_left + par_width
    x_btm_left = x_btm_right - par_width

    left_line = set_linear_equation(upper_left, (x_btm_left, y_btm))
    right_line = set_linear_equation(bottom_right, (x_up_right, y_up))
    points = []

    for coordinate in [min(x_btm_left,x_up_left), max(x_btm_right,x_up_right)]:
        if coordinate is height or coordinate < 0:
            print('coordinate ' + str(coordinate) + ' out of bound!')
            return []

    for coordinate in [y_up, y_up]:
        if coordinate is width or coordinate < 0:
            print('coordinate ' +  str(coordinate) + ' out of bound!')            
            return []


    for y in range(y_up, y_btm):
        for x in range(min(x_btm_left,x_up_left),max(x_btm_right,x_up_right) ):
            try:
                if img[y, x] == 0:
                    continue
            except:
                pass
            else:
                if min(left_line(x),right_line(x)) <= y <= max(left_line(x),right_line(x)):
                    points.append((x, y))

    return points
