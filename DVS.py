import cv2
from DVSUtils import get_data_from_image, delete_none_binary_pixels, delete_noise_by_neighbors, plot_by_points, \
    hough_line_transform

img1 = cv2.imread('10.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# hough_line_transform(img1)
delete_noise_by_neighbors(img1)
# img1 = delete_noise_by_neighbors(img1,min_neighbors_amount=2)
# img1 = delete_noise_by_neighbors(img1,min_neighbors_amount=1)
# img1 = set_triangle_scope(img1)
# img1 = delete_right_half(img1)



#
# cv2.imshow('1', img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
