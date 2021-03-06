from typing import List
import cv2
import numpy as np
from matplotlib import pyplot as plt

from sklearn import linear_model
from Utils.Plot import draw_parallelogram, plot_data, show_image
from Utils.Preprocess import general, get_data_from_parallelogram, get_params_for_linear_equation
from Utils.Types import Point


def get_par_from_RANSAC_line(RANSAC_line_by_y, old_y_up: int, par_width: int, par_height: int):
    mid_btm_x = RANSAC_line_by_y(old_y_up)
    new_y_up = old_y_up - par_height

    mid_up_x = RANSAC_line_by_y(new_y_up)
    new_bottom_right = int(mid_btm_x + 0.5 * par_width), old_y_up
    new_upper_left = int(mid_up_x - 0.5 * par_width), new_y_up
    return new_upper_left, new_bottom_right


def get_points_from_shape(img: np.ndarray, min_y, max_y, min_x, max_x, max_line, min_line) -> List[Point]:
    points = []
    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            if img[y, x] == 0:
                continue
            elif min_line(x) <= y <= max_line(x):
                points.append((x, y))
    return points


def get_init_data(img: np.ndarray, par_height: int):
    height, width = img.shape

    m = -1.28
    n_1 = 650
    n_2 = 900

    def y_1(x): return m * x + n_1
    def x_2(y): return (y - n_2) / m
    def y_2(x): return m * x + n_2
    def x_1(y): return (y - n_1) / m

    upper_left = (int(x_1(height - par_height)), height - par_height)
    upper_right = (int(x_2(height - par_height)), height - par_height)
    bottom_right = (int(x_2(799)), 799)
    bottom_left = (0, int(y_1(0)))

    left_points = get_points_from_shape(
        img, height - par_height, height, 0, int(x_2(height - par_height)), y_2, y_1)

    cv2.line(img, upper_left, upper_right, 255, 1)
    cv2.line(img, upper_left, bottom_left, 255, 1)
    cv2.line(img, upper_right, bottom_right, 255, 1)
    
   ####################################################

    m = 1.28
    n_1 = -700
    n_2 = -950

    def x_1(y): return (y - n_1) / m
    def x_2(y): return (y - n_2) / m
    def y_1(x): return m * x + n_1
    def y_2(x): return m * x + n_2

    upper_left = (int(x_1(height - par_height)), height - par_height)
    bottom_left = (int(x_1(799)), 799)
    upper_right = (int(x_2(height - par_height)), height - par_height)
    bottom_right = (width, int(y_2(width)))

    right_points = get_points_from_shape(
        img, height - par_height, height, int(x_1(height - par_height)), width, y_1, y_2)

    cv2.line(img, upper_left, upper_right, 255, 1)
    cv2.line(img, upper_left, bottom_left, 255, 1)
    cv2.line(img, upper_right, bottom_right, 255, 1)

    return left_points, right_points


def main_par_regression_loop(img_path='images/10.jpg', par_height=300, par_width=100):

    img = general(img_path, min_neighbors_amount_list=[2, 1])

    first_y = 500

    left_points, right_points = get_init_data(img, par_height)

    left_par_list, left_estimated_points = calc_pars(
        img, left_points, par_width, par_height, first_y)

    right_par_list, right_estimated_points = calc_pars(
        img, right_points,   par_width, par_height, first_y)

    for par in left_par_list:
        upper_left, bottom_right, par_width, par_height = par
        draw_parallelogram(img, (upper_left, bottom_right, par_width))

    for par in right_par_list:
        upper_left, bottom_right, par_width, par_height = par
        draw_parallelogram(img, (upper_left, bottom_right, par_width))
    show_image(img)

    print('Left estimated points:', left_estimated_points)
    print('Right estimated points:', right_estimated_points)
    return left_estimated_points, right_estimated_points


def get_par_from_points(par_width,  par_height, points, old_y):
    m, n = line_params_from_RANSAC(points)

    def linear_equation_by_y(y): return ((y-n) / m)

    new_upper_left, new_bottom_right = get_par_from_RANSAC_line(
        linear_equation_by_y, old_y, par_width, par_height)

    new_par = (new_upper_left, new_bottom_right, par_width, par_height)

    return new_par, (int(linear_equation_by_y(500)), old_y), (m, n)


def calc_pars(img: np.ndarray, first_points,  par_width: int, par_height: int, old_y: int, total_par_amount=2):
    par_width_delta = 0.7
    par_height_delta = 0.5
    first_calc_par, first_estimated_point, line_params = get_par_from_points(
        int(par_width * par_width_delta),  int(par_height * par_height_delta), first_points, old_y)

    line_list = [line_params]
    par_list = [first_calc_par]
    estimated_points = [first_estimated_point]
    for i in range(total_par_amount - 1):

        old_par = par_list[-1]

        upper_left, bottom_right, current_par_width, current_par_height = old_par
        _, y_up = upper_left

        points = get_data_from_parallelogram(
            img, (upper_left, bottom_right, current_par_width))
        if len(points) is 0:
            print('len(points) is 0:')
            m, n = line_list[-1]
        else:
            m, n = line_params_from_RANSAC(points)

        def linear_equation_by_y(y): return ((y-n) / m)

        new_par_width = int(current_par_height * par_width_delta)
        new_par_height = int(current_par_width * par_height_delta)

        new_upper_left, new_bottom_right = get_par_from_RANSAC_line(
            linear_equation_by_y, y_up, new_par_width, new_par_height)

        estimated_points.append((int(linear_equation_by_y(y_up)), y_up))
        par_list.append((new_upper_left, new_bottom_right,
                        new_par_width, new_par_height))
    #  add last point
    estimated_points.append(
        (int(linear_equation_by_y(new_upper_left[1])), new_upper_left[1]))

    return par_list, estimated_points


def line_params_from_RANSAC(points):
    # plot_RANSAC(points)
    x, y = zip(*points)

    y = np.asarray(y)
    x = np.array(x)[:, np.newaxis]

    ransac = linear_model.RANSACRegressor()
    ransac.fit(x, y)

    line_x = np.arange(x.min(), x.max())[:, np.newaxis]
    line_y = ransac.predict(line_x)
    y_0 = line_y[0]
    y_1 = line_y[1]

    x_0 = np.asscalar(line_x[0])
    x_1 = np.asscalar(line_x[1])

    m, n = get_params_for_linear_equation((x_1, y_1), (x_0, y_0))

    return m, n


def plot_RANSAC(points):
    # points = image_to_data(img)
    x, y = zip(*points)
    y = np.asarray(y)
    X = np.array(x)[:, np.newaxis]

    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)
    inliner_mask = ransac.inlier_mask_
    outliner_mask = np.logical_not(inliner_mask)

    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)

    plt.scatter(
        X[inliner_mask], y[inliner_mask], color="yellowgreen", marker=".", label="Inliers"
    )
    plt.scatter(
        X[outliner_mask], y[outliner_mask], color="gold", marker=".", label="Outliers"
    )
    plt.plot(
        line_X,
        line_y_ransac,
        color="cornflowerblue",
        linewidth=2,
        label="RANSAC regressor",
    )
    plt.legend(loc="lower right")
    plt.xlabel("Input")
    plt.ylabel("Response")
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.xaxis.tick_top()
    plt.show()


def points_close_to_mean(points: List[Point], threshold=30) -> bool:
    x, y = zip(*points)

    mean = sum(y)/len(y)
    close_to_mean = len(filter(lambda i: abs(i-mean) <= threshold, y))

    close_percent = close_to_mean/len(y)
    return close_percent > 0.8
