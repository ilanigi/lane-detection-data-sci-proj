import numpy as np
from matplotlib import pyplot as plt

from sklearn import linear_model

from src.Utils.Crop import crop_rectangle
from src.Utils.Plot import show_image, plot_data
from src.Utils.Preprocess import image_to_data


def get_point_from_RANSAC(img, current_rectangle):
    upper_left, bottom_right = current_rectangle
    x_left, y_left = upper_left

    cropped_img = crop_rectangle(img, current_rectangle)
    height, width = cropped_img.shape
    # show_image(cropped_img)
    m, local_x, local_y = RANSAC(cropped_img)
    local_y = height - local_y
    global_x = x_left + local_x
    global_y = y_left + local_y
    n = global_y - m * global_x
    new_x = (y_left - n) / m

    return int(new_x), y_left


def RANSAC(img):
    plot_RANSAC(img)

    points = image_to_data(img)
    plot_data(points)
    X, y = zip(*points)
    y = np.asarray(y)
    X = np.array([[point] for point in X])

    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)

    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)
    max_y = np.asscalar(max(line_y_ransac))
    max_index = int(np.where(line_y_ransac == max_y))
    max_x = np.asscalar(line_X[max_index])
    min_index = 0

    if max_index == 0:
        min_index = -1

    min_x = np.asscalar(line_X[min_index])
    min_y = np.asscalar(line_y_ransac[min_index])
    m = (min_y - max_y) / (min_x - max_x)

    return m, max_x, max_y


def plot_RANSAC(img):
    points = image_to_data(img)
    y, X = zip(*points)
    y = np.asarray(y)
    X = np.array([[point] for point in X])

    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)
    inliner_mask = ransac.inlier_mask_
    outliner_mask = np.logical_not(inliner_mask)

    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)

    lw = 2
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
        linewidth=lw,
        label="RANSAC regressor",
    )
    plt.legend(loc="lower right")
    plt.xlabel("Input")
    plt.ylabel("Response")
    plt.show()
