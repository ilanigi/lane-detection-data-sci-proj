import cv2

from Utils.Hough import all_hough_circle_transform

img1 = cv2.imread('142.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

all_hough_circle_transform(img1)
# delete_noise_by_neighbors(img1)
# img1 = set_triangle_scope(img1)
# img1 = delete_right_half(img1)



#
# cv2.imshow('1', img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
