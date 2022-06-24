import cv2
import numpy as np
from matplotlib import pyplot as plt
# from DVSUtils import set_triangle_scope, delete_none_binary_pixels, delete_noise_by_neighbors, delete_right_half
from DVSUtils import get_data_from_image, delete_none_binary_pixels, delete_noise_by_neighbors, set_triangle_scope, \
    plot_by_points

img1 = cv2.imread('10.jpg')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
height, width = img1.shape
img1 = delete_none_binary_pixels(img1)
img1 = delete_noise_by_neighbors(img1,min_neighbors_amount=2)
img1 = delete_noise_by_neighbors(img1,min_neighbors_amount=1)
plot_by_points(img1)
# img1 = set_triangle_scope(img1)
# img1 = delete_right_half(img1)
# x,y = get_data_from_image(img1)
# plt.plot(x, y)

# plt.plot(img1)
# plt.ylabel('some numbers')
# plt.show()
#
# cv2.imshow('1', img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
