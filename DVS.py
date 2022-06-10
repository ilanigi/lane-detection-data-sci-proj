import cv2
# from cv2 import Canny
# from cv2 import line
import numpy as np

from DVSUtils import deleteNoiseByNeighbors, deleteNoneBinaryPixels
# import math


img1 = cv2.imread('10.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

hight, width  = img1.shape

img1 = deleteNoneBinaryPixels(img1)
img1 = deleteNoiseByNeighbors(img1,minNeighborAmount=2)


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

    
