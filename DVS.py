import cv2
from cv2 import Canny


img1 = cv2.imread('10.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

hight, width  = img1.shape

half = 255/2

for i in range(hight):
    for j in range(width):
        if img1[i,j] > half:
            img1[i,j] = 255
        else:
            img1[i,j] = 0


img1 = img1[240:,0:650]

cv2.imshow('1', img1)

cv2.waitKey(0)
cv2.destroyAllWindows()

img1 = img1/255

