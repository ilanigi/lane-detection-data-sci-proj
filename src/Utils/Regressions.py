import numpy as np
from matplotlib import pyplot as plt

from sklearn import linear_model

from src.Utils.Crop import crop_rectangle
from src.Utils.Plot import get_rectangle_from_mid_bottom, draw_rectangle, show_image, draw_parallelogram
from src.Utils.Preprocess import general, get_data_from_parallelogram, set_linear_equation, set_linear_equation_by_y


def get_par_from_mid_point(RANSAC_line_by_x, old_y_up, par_width, par_height):
    mid_btm_x = RANSAC_line_by_x(old_y_up)

    new_y_up = old_y_up - par_height

    mid_up_x = RANSAC_line_by_x(new_y_up)
    new_bottom_right = int(mid_btm_x + 0.5 * par_width), old_y_up
    new_upper_left = int(mid_up_x - 0.5 * par_width), new_y_up
    return new_upper_left, new_bottom_right


# deprecated
def par_regression_loop(img_path = 'images/10.jpg', par_height = 120, par_width = 120, par_amount=3):
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

        points = get_data_from_parallelogram(img, (upper_left, bottom_right, par_height))
        m, x_on_reg_line, y_on_reg_line = line_from_RANSAC(points)
        upper_left, bottom_right = get_par_from_mid_point(m, x_on_reg_line,y_on_reg_line,par_width,par_height, x_left)
        right_pars.append((upper_left, bottom_right, par_height))

        upper_left, bottom_right, par_height = left_pars[-1]
        x_left, y_up_left = upper_left

        points = get_data_from_parallelogram(img, (upper_left, bottom_right, par_height))
        m, x_on_reg_line, y_on_reg_line = line_from_RANSAC(points)
        upper_left, bottom_right = get_par_from_mid_point(m, x_on_reg_line, y_on_reg_line, par_width, par_height,
                                                          x_left)
        left_pars.append((upper_left, bottom_right, par_height))

    for par in right_pars:

        draw_parallelogram(img,par)

    for par in left_pars:
        draw_parallelogram(img,par)



def rac_regression_loop(img_path = 'images/10.jpg', rectangle_height = 150, rectangle_width = 250, rec_amount=4):

    img = general(img_path)

    height, width = img.shape

    right_rec_list = [((width - rectangle_width, height - rectangle_height), (width - 1, height - 1))]
    left_rec_list = [((0, height - rectangle_height), (rectangle_width, height - 1))]

    for i in range(rec_amount):
        if i >= 1:
            rectangle_height /= 1.5
            rectangle_width /= 1.5

        left_rec = left_rec_list[-1]
        point = get_point_from_RANSAC(img, left_rec)
        left_rec = get_rectangle_from_mid_bottom(point, rectangle_width, rectangle_height, width)
        left_rec_list.append(left_rec)

        right_rec = right_rec_list[-1]
        point = get_point_from_RANSAC(img, right_rec)
        right_rec = get_rectangle_from_mid_bottom(point, rectangle_width, rectangle_height, width)
        right_rec_list.append(right_rec)

    for rec in right_rec_list:
        draw_rectangle(img, rec)

    for rec in left_rec_list:
        draw_rectangle(img, rec)

    show_image(img)


def par_main_regression_loop(img_path = 'images/142.jpg'):
    par_height = 280
    par_width = 120
    img = general(img_path, min_neighbors_amount_list=[1])

    height, width = img.shape


    y_up = height - par_height
    y_btm = height - 1
    x_btm_right = par_width
    x_up_left = 180
    upper_left = x_up_left, y_up
    bottom_right = x_btm_right, y_btm

    # # x_left = width - 200
    # # y_up_left = height - 140 - length
    # # x_right = width
    # # y_btm_right = height
    # upper_left = x_left, y_up_left
    # bottom_right = x_right, y_btm_right

    points = get_data_from_parallelogram(img, (upper_left, bottom_right, par_width))
    x = line_from_RANSAC(points)
    plot_data(points)
    #
    new_par_height = int(par_height / 4)
    new_upper_left, new_bottom_right = get_par_from_mid_point(x, y_up, par_width, new_par_height)
    #
    draw_parallelogram(img, (new_upper_left, new_bottom_right, par_width))
    draw_parallelogram(img, (upper_left, bottom_right, par_width))
    show_image(img)


def get_point_from_RANSAC(img, current_rectangle):
    upper_left, bottom_right = current_rectangle
    x_left, y_left = upper_left

    cropped_img = crop_rectangle(img, current_rectangle)

    m, local_x, local_y = line_from_RANSAC(cropped_img)

    global_x = x_left + local_x
    global_y = y_left + local_y
    n = global_y - m * global_x
    new_x = (y_left - n) / m

    return int(new_x), y_left


def line_from_RANSAC(points):
    plot_RANSAC(points)

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

    x = set_linear_equation_by_y((x_1, y_1), (x_0, y_0))

    return x


def plot_RANSAC(points):
    # points = image_to_data(img)
    y, X = zip(*points)
    y = np.asarray(y)
    X = np.array([[point] for point in X])

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
