import cv2
from Utils.Crop import get_bottom_right_corner
from Utils.Preprocess import general
from Utils.Regressions import RANSAC


def main():
    img1 = general('10.jpg')
    img1 = get_bottom_right_corner(img1)
    RANSAC(img1)


if __name__ == '__main__':
    main()
