import cv2
from Utils.Linear import get_bottom_right_corner
from Utils.Preprocess import delete_none_binary_pixels, delete_noise_by_neighbors, get_data_from_image, plot_by_points, \
    general

img1 = general('10.jpg')
img1 = get_bottom_right_corner(img1)
# points = get_data_from_image(img1)

plot_by_points(img1)
