import numpy as np
from matplotlib import pyplot as plt

from sklearn import linear_model, datasets

from Utils.Preprocess import image_to_data, plot_data, general


def RANSAC(img):
    points = image_to_data(img)
    X, y = zip(*points)
    plot_data(points)
    y = np.asarray(y)
    X = np.array([[point] for point in X])

    lr = linear_model.LinearRegression()
    lr.fit(X, y)

    # Robustly fit linear model with RANSAC algorithm
    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Predict data of estimated models
    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y = lr.predict(line_X)
    line_y_ransac = ransac.predict(line_X)

    lw = 2
    plt.scatter(
        X[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
    )
    plt.scatter(
        X[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers"
    )
    plt.plot(line_X, line_y, color="navy", linewidth=lw, label="Linear regressor")
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
