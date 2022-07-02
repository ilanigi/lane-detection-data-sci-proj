import numpy as np
from cv2.cv2 import imread, cvtColor, COLOR_BGR2GRAY


def general(image_name, min_neighbors_amount_list = [2, 1]):
    img = imread(image_name)
    img = cvtColor(img, COLOR_BGR2GRAY)
    img = delete_none_binary_pixels(img)
    for min_neighbor in min_neighbors_amount_list:
        img = delete_noise_by_neighbors(img, min_neighbors_amount=min_neighbor)
    return img


def delete_noise_by_neighbors(img, kernel=[-1, 0, 1], min_neighbors_amount=3):
    height, width = img.shape
    out_img = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if img[i, j] == 0:
                continue
            counter = 0
            for k in kernel:
                for m in kernel:
                    try:
                        if img[i + k, j + m] == 255:
                            counter += 1
                    except IndexError:
                        # kernel out of picture bound
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


def image_to_data(img):
    points = []
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            if img[i, j] == 0:
                continue
            else:
                points.append((j, i))

    return points
