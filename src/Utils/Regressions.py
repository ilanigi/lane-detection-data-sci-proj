import numpy as np
from matplotlib import pyplot as plt

from sklearn import linear_model

from src.Utils.Crop import crop_rectangle
from src.Utils.Plot import get_rectangle_from_mid_bottom, draw_rectangle, show_image, draw_parallelogram
from src.Utils.Preprocess import general, get_data_from_parallelogram


def get_par_from_mid_point(m, x_on_reg_line, y_on_reg_line, par_width, par_height, old_x, direction='left'):
    n = y_on_reg_line - m * x_on_reg_line

    def y(x): return m * x + n

    y_mid_point = y(old_x)
    if direction is 'left':
        new_x_left = old_x
        y_up_left = y_mid_point - 0.5 * par_height
        new_x_right = new_x_left + par_width
        y_btm_right = y(new_x_right) + 0.5 * par_height

        upper_left = int(new_x_left), int(y_up_left)
        bottom_right = int(new_x_right), int(y_btm_right)
        return upper_left,bottom_right
    else:
        new_x_right = old_x
        new_x_left = new_x_right - par_width
        y_btm_right = y_mid_point + 0.5 * par_height
        y_up_left = y(new_x_left) - 0.5 * par_height

        upper_left = int(new_x_left), int(y_up_left)
        bottom_right = int(new_x_right), int(y_btm_right)
        return upper_left, bottom_right


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
        m, x_on_reg_line, y_on_reg_line = RANSAC(points)
        upper_left, bottom_right = get_par_from_mid_point(m, x_on_reg_line,y_on_reg_line,par_width,par_height, x_left)
        right_pars.append((upper_left, bottom_right, par_height))

        upper_left, bottom_right, par_height = left_pars[-1]
        x_left, y_up_left = upper_left

        points = get_data_from_parallelogram(img, (upper_left, bottom_right, par_height))
        m, x_on_reg_line, y_on_reg_line = RANSAC(points)
        upper_left, bottom_right = get_par_from_mid_point(m, x_on_reg_line, y_on_reg_line, par_width, par_height,
                                                          x_left)
        left_pars.append((upper_left, bottom_right, par_height))

    for par in right_pars:

        draw_parallelogram(img,par)

    for par in left_pars:
        draw_parallelogram(img,par        )



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


def get_point_from_RANSAC(img, current_rectangle):
    upper_left, bottom_right = current_rectangle
    x_left, y_left = upper_left

    cropped_img = crop_rectangle(img, current_rectangle)

    m, local_x, local_y = RANSAC(cropped_img)

    global_x = x_left + local_x
    global_y = y_left + local_y
    n = global_y - m * global_x
    new_x = (y_left - n) / m

    return int(new_x), y_left


def RANSAC(points):
    plot_RANSAC(points)

    # points = image_to_data(img)
    # plot_data(points)
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

    m = (y_1 - y_0) / (x_1 - x_0)

    return m, x_1, y_1


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
