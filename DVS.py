import cv2

from DVSUtils import set_triangle_scope, delete_none_binary_pixels, delete_noise_by_neighbors

img1 = cv2.imread('142.jpg')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
height, width = img1.shape
img1 = delete_none_binary_pixels(img1)
# img1 = delete_noise_by_neighbors(img1,min_neighbors_amount=1)
mid_point = (int(width / 2), int(height*0.4))
img1 = set_triangle_scope(img1, mid_point)

cv2.imshow('1', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
