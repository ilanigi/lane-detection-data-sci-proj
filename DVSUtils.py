import numpy as np
from skimage.transform import hough_line, hough_line_peaks
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

    img_unit8 = np.uint8(img)
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180)
    hspace, theta, dist = hough_line(img_unit8, tested_angles)
    angle_list = []
    fig, axes = plt.subplots(1, 1)



    axes.imshow(np.log(1 + hspace), aspect=0.05)
    plt.show()
    fig, axes = plt.subplots(1, 1)
    axes.imshow(img, cmap='gray')

    origin = np.array((0, img.shape[1]))
    min_angels = [(9999,0), (-9999,0)]
    for _, angle, dist in zip(*hough_line_peaks(hspace, theta, dist)):
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
    axes.set_title('Detected lines')

    plt.show()


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
