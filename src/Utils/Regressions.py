import numpy as np
from matplotlib import pyplot as plt

from sklearn import linear_model

from src.Utils.Preprocess import image_to_data


def RANSAC(img):
    points = image_to_data(img)
    X, y = zip(*points)

    y = np.asarray(y)
    X = np.array([[point] for point in X])

    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)

    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)
    print(line_y_ransac)


def plot_RANSAC(img):
    points = image_to_data(img)
    X, y = zip(*points)
    y = np.asarray(y)
    X = np.array([[point] for point in X])

    # Robustly fit linear model with RANSAC algorithm
    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)
    inliner_mask = ransac.inlier_mask_
    outliner_mask = np.logical_not(inliner_mask)

    # Predict data of estimated models
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
