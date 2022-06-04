
import cv2
from cv2 import Canny


img1 = cv2.imread('00630.jpg')
img2 = cv2.imread('00660.jpg')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

img1 = cv2.GaussianBlur(img1,(3,3),2)
img2 = cv2.GaussianBlur(img2,(3,3),2)


delta = img1 - img2*0.9
print(delta)

# blur = cv2.GaussianBlur(gray,(3,3),2)

# canny = Canny(blur,27,170)

cv2.imshow('Canny image', delta)


  
cv2.waitKey(0)
cv2.destroyAllWindows()