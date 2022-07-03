import numpy as np
from cv2.cv2 import imshow, waitKey, destroyAllWindows
from matplotlib import pyplot as plt


def get_rectangle_from_mid_bottom(mid_bottom_point, length, height, img_width):
    x, y = mid_bottom_point
    if 2*x < length:
        x = length/2
    elif 2*x + length > 2*img_width:
        x = img_width - length/2 - 1
    return (int(x - length / 2), int(y - height)), (int(x + length / 2), y)


def draw_rectangle(img, rectangle):
    height, width = img.shape
    upper_left, bottom_right = rectangle
    x_left, y_left = upper_left
    x_right, y_right = bottom_right

    for coordinate in [x_left, x_right]:
        if coordinate is width or coordinate < 0:
            raise Exception("rectangle out of bound")

    for coordinate in [y_left, y_right]:
        if coordinate is height or coordinate < 0:
            raise Exception("rectangle out of bound")

    if x_left > x_right or y_left > y_right:
        raise Exception("rectangle is misplaced")

    img[y_right, x_left:x_right] = 255
    img[y_left, x_left:x_right] = 255

    img[y_left:y_right, x_left] = 255
    img[y_left:y_right, x_right] = 255


def show_image(img, img_name='img'):
    imshow(img_name, img)
    waitKey(0)
    destroyAllWindows()


def plot_data(points):
    x, y = zip(*points)
    plt.scatter(x, y, s=0.5, c='k')

    ax = plt.gca()  # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.xaxis.tick_top()  # and move the X-Axis
    plt.show()
