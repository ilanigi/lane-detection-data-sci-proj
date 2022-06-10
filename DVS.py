import cv2
# from cv2 import Canny
# from cv2 import line
import numpy as np

from DVSUtils import delete_noise_by_neighbors, delete_none_binary_pixels
# import math

img1 = cv2.imread('10.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

height, width = img1.shape

img1 = delete_none_binary_pixels(img1)
img1 = delete_noise_by_neighbors(img1, min_neighbors_amount=2)


# img1 = cv2.GaussianBlur(img1,(7,7),21)

cv2.imshow('1', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()


# img1 = img1[240:,0:650]

    
# linesP = cv2.HoughLinesP(img1, 1, np.pi / 180, 50, None, 50, 10)

# if linesP is not None:
#     for i in range(0, len(linesP)):
#         l = linesP[i][0]
#         cv2.line(img1, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)


cv2.imshow('1', img1)

cv2.waitKey(0)
cv2.destroyAllWindows()




# img1 = img1/255

    
