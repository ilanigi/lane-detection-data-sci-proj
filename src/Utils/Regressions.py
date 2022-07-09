from ast import Lambda
from typing import List
import cv2
from cv2 import mean
import numpy as np
from matplotlib import pyplot as plt

from sklearn import linear_model
from Utils.Plot import draw_parallelogram, get_rectangle_from_mid_bottom, plot_data, show_image
from Utils.Preprocess import general, get_data_from_parallelogram, get_params_for_linear_equation
from Utils.Types import Point


def get_par_from_mid_point(RANSAC_line_by_y:Lambda, old_y_up:int, par_width:int, par_height:int):
    mid_btm_x = RANSAC_line_by_y(old_y_up)

    new_y_up = old_y_up - par_height

    mid_up_x = RANSAC_line_by_y(new_y_up)
    new_bottom_right = int(mid_btm_x + 0.5 * par_width), old_y_up
    new_upper_left = int(mid_up_x - 0.5 * par_width), new_y_up
    return new_upper_left, new_bottom_right


# deprecated
def par_regression_loop(img_path='images/10.jpg', par_height=120, par_width=120, par_amount=3):
    img = general(img_path)

    height, width = img.shape

    x_left = 0
    y_up_left = height - par_height
    x_right = par_width
    y_btm_right = height - 70
    upper_left = x_left, y_up_left
    bottom_right = x_right, y_btm_right
    left_pars = [(upper_left, bottom_right, par_height)]

    x_left = width - 200
    y_up_left = height - 140 - par_height
    x_right = width
    y_btm_right = height
    upper_left = x_left, y_up_left
    bottom_right = x_right, y_btm_right
    right_pars = [(upper_left, bottom_right, par_height)]

    for i in range(par_amount):
        upper_left, bottom_right = right_pars[-1]
        x_left, y_up_left = upper_left

        points = get_data_from_parallelogram(
            img, (upper_left, bottom_right, par_height))
        m, x_on_reg_line, y_on_reg_line = line_params_from_RANSAC(points)

        upper_left, bottom_right = get_par_from_mid_point(
            m, x_on_reg_line, y_on_reg_line, par_width, par_height, x_left)
        right_pars.append((upper_left, bottom_right, par_height))

        upper_left, bottom_right, par_height = left_pars[-1]
        x_left, y_up_left = upper_left

        points = get_data_from_parallelogram(
            img, (upper_left, bottom_right, par_height))
        m, x_on_reg_line, y_on_reg_line = line_params_from_RANSAC(points)
        upper_left, bottom_right = get_par_from_mid_point(m, x_on_reg_line, y_on_reg_line, par_width, par_height,
                                                          x_left)
        left_pars.append((upper_left, bottom_right, par_height))

    for par in right_pars:

        draw_parallelogram(img, par)

    for par in left_pars:
        draw_parallelogram(img, par)


def rac_regression_loop(img_path='images/10.jpg', rectangle_height=150, rectangle_width=250, rec_amount=4):

    img = general(img_path)

    height, width = img.shape

    right_rec_list = [((width - rectangle_width, height -
                       rectangle_height), (width - 1, height - 1))]
    left_rec_list = [((0, height - rectangle_height),
                      (rectangle_width, height - 1))]

    for i in range(rec_amount):
        if i >= 1:
            rectangle_height /= 1.5
            rectangle_width /= 1.5

        left_rec = left_rec_list[-1]
        point = get_point_from_RANSAC(img, left_rec)
        left_rec = get_rectangle_from_mid_bottom(
            point, rectangle_width, rectangle_height, width)
        left_rec_list.append(left_rec)

        right_rec = right_rec_list[-1]
        point = get_point_from_RANSAC(img, right_rec)
        right_rec = get_rectangle_from_mid_bottom(
            point, rectangle_width, rectangle_height, width)
        right_rec_list.append(right_rec)

    for rec in right_rec_list:
        draw_rectangle(img, rec)

    for rec in left_rec_list:
        draw_rectangle(img, rec)

    show_image(img)


def main_par_regression_loop(img_path='images/142.jpg'):

    par_height = 180
    par_width = 120
    img = general(img_path, min_neighbors_amount_list=[2,1])

    height, width = img.shape

    upper_left = (180, height - par_height)
    bottom_right = (par_width, height - 1)

    left_par_list, left_estimated_points = calc_pars(
        img, upper_left, bottom_right, par_width, par_height)

    upper_left = (width - 180 - par_width, height - par_height)
    bottom_right = (par_width-1, height - 1)
    right_par_list, right_estimated_points = calc_pars(
        img, upper_left, bottom_right, par_width, par_height)

    for par in left_par_list:
        upper_left, bottom_right, par_width, par_height = par
        draw_parallelogram(img, (upper_left, bottom_right, par_width))

    for par in right_par_list:
        upper_left, bottom_right, par_width, par_height = par
        draw_parallelogram(img, (upper_left, bottom_right, par_width))
    show_image(img)

    print('Left estimated points:', left_estimated_points)
    print('Right estimated points:', right_estimated_points)


def calc_pars(img: np.ndarray, upper_left:Point, bottom_right:Point ,par_width: int, par_height: int, par_amount=1):
    par_list = [(upper_left, bottom_right, par_width, par_height)]
    estimated_points = []
    for i in range(par_amount):

        old_par = par_list[-1]

        upper_left, bottom_right, par_width, par_height = old_par
        x_up, y_up = upper_left

        draw_parallelogram(img,old_par)
        show_image(img)

        points = get_data_from_parallelogram(
            img, (upper_left, bottom_right, par_width))

        # plot_data(points)

        m,n = line_params_from_RANSAC(points)
        
        linear_equation_by_y = lambda y:((y-n) / m)
        linear_equation_by_x = lambda x:(m*x + n)

        
        xy = [(i,int(linear_equation_by_x(i))) for i in range(600) if 0 < int(linear_equation_by_x(i)) < 800]

        max:tuple
        min:tuple
        min_y = 9999
        max_y = -1
        for point in xy:
            x,y = point
            if y<min_y: 
                min = point
                min_y = y
            if y>max_y:
                max = point
                max_y = y

        cv2.line(img, min, max, 255, 1)
        show_image(img)


        # estimated_points.append((int(linear_equation(y_up)), y_up))

        # new_par_height = int(par_height * 2 / 3)

        # new_upper_left, new_bottom_right = get_par_from_mid_point(
        #     linear_equation, y_up, par_width, new_par_height)
        # par_list.append((new_upper_left, new_bottom_right,
        #                 par_width, new_par_height))

    return par_list, estimated_points


def get_point_from_RANSAC(img, current_rectangle):
    upper_left, bottom_right = current_rectangle
    x_left, y_left = upper_left

    cropped_img = crop_rectangle(img, current_rectangle)

    m, local_x, local_y = line_params_from_RANSAC(cropped_img)

    global_x = x_left + local_x
    global_y = y_left + local_y
    n = global_y - m * global_x
    new_x = (y_left - n) / m

    return int(new_x), y_left

def points_close_to_mean(points:List[Point], threshold = 15)->bool:
    x, y = zip(*points)
    
    mean = sum(y)/len(y)
    close_to_mean = len(filter(lambda i:abs(i-mean) <= threshold, y))
    
    close_percent = close_to_mean/len(y)
    return close_percent > 0.8


def line_params_from_RANSAC(points):
    plot_RANSAC(points)
    x, y = zip(*points)

    if points_close_to_mean(points):
        y = np.asarray(y)
        x = np.array(x)[:, np.newaxis]

        mean_y = sum(y)/len(y)
        mean_x = sum(y)/len(y)
        close_points = filter(lambda point: abs(mean_y - mean_y[1]) < 15 and abs(mean_x - mean_y[1])< 15, points)
        x, y = zip(*close_points)
        max_y = max(y)
        min_y=9999
        max_x=0
        min_x=9999

        fot


        

       
    else:
        
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
    x,y = zip(*points)
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
