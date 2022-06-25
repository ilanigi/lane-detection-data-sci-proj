import numpy as np
from matplotlib import pyplot as plt
from skimage.transform._hough_transform import circle_perimeter
from skimage.transform import hough_line, hough_line_peaks, hough_circle, hough_circle_peaks

from Utils.Preprocess import delete_noise_by_neighbors, delete_none_binary_pixels


def all_hough_circle_transform(img):
    img = delete_none_binary_pixels(img)
    img = delete_noise_by_neighbors(img, min_neighbors_amount=2)

    img_unit8 = np.uint8(img)
    radius = np.arange(450, 500)
    h_space = hough_circle(img_unit8, radius)
    accum, cx, cy, rad = hough_circle_peaks(h_space, radius)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    counter = 0
    for center_y, center_x, radius in zip(cy, cx, rad):
        if counter == 100:
            break
        counter += 1
        circ_y, circ_x = circle_perimeter(center_y, center_x, radius, shape=img.shape)
        img[circ_y, circ_x] = 225

    ax.imshow(img, cmap=plt.cm.gray)
    plt.show()


def all_hough_lines_transform(img):
    img = delete_none_binary_pixels(img)
    img = delete_noise_by_neighbors(img, min_neighbors_amount=2)

    img_unit8 = np.uint8(img)
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)
    h_space, theta, dist = hough_line(img_unit8, tested_angles)
    angle_list = []
    fig, axes = plt.subplots(1, 1)

    axes.imshow(np.log(1 + h_space), aspect=0.05)
    plt.show()
    fig, axes = plt.subplots(1, 1)
    axes.imshow(img, cmap='gray')

    origin = np.array((0, img.shape[1]))
    for _, angle, dist in zip(*hough_line_peaks(h_space, theta, dist)):
        angle_list.append(angle)
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        axes.plot(origin, (y0, y1), '-r')
    axes.set_xlim(origin)
    axes.set_ylim((img.shape[0], 0))
    axes.set_axis_off()
    axes.set_title('Detected lines')

    plt.show()


def hough_line_transform(img):
    img = delete_none_binary_pixels(img)
    img = delete_noise_by_neighbors(img, min_neighbors_amount=2)

    img_unit8 = np.uint8(img)
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)
    h_space, theta, dist = hough_line(img_unit8, tested_angles)
    angle_list = []
    fig, axes = plt.subplots(1, 1)

    axes.imshow(np.log(1 + h_space), aspect=0.05)
    plt.show()
    fig, axes = plt.subplots(1, 1)
    axes.imshow(img, cmap='gray')

    origin = np.array((0, img.shape[1]))
    min_angels = [(9999, 0), (-9999, 0)]
    for _, angle, dist in zip(*hough_line_peaks(h_space, theta, dist)):
        if abs(angle) < 0.1:
            continue

        if 0 < angle < min_angels[0][0]:
            min_angels[0] = (angle, dist)
        elif min_angels[1][0] < angle < 0 and abs(abs(min_angels[0][0]) - abs(angle)) > 0.1:
            min_angels[1] = (angle, dist)

    for angle, dist in min_angels:
        angle_list.append(angle)
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        axes.plot(origin, (y0, y1), '-r')
    axes.set_xlim(origin)
    axes.set_ylim((img.shape[0], 0))
    axes.set_axis_off()
    axes.set_title('Min detected lines')

    plt.show()
