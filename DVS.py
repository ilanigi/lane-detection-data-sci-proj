import cv2
from Utils.Linear import get_bottom_right_corner
from Utils.Preprocess import delete_none_binary_pixels, delete_noise_by_neighbors, get_data_from_image, plot_by_points

img1 = cv2.imread('10.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1 = delete_none_binary_pixels(img1)
img1 = delete_noise_by_neighbors(img1)
img1 = delete_noise_by_neighbors(img1, min_neighbors_amount=1)
img1 = get_bottom_right_corner(img1)
# points = get_data_from_image(img1)

plot_by_points(img1)
