from Utils.Preprocess import general
from src.Utils.Plot import draw_rectangle, show_image, get_rectangle_from_mid_bottom
from src.Utils.Regressions import get_point_from_RANSAC, RANSAC


def main():
    rectangle_height = 100
    rectangle_width = 200
    img_path = 'images/10.jpg'
    img = general(img_path)

    height, width = img.shape
    left_rec = ((0, height - rectangle_height), (rectangle_width, height - 1))
    right_rec = ((width - rectangle_width, height - rectangle_height), (width - 1, height - 1))

    point = get_point_from_RANSAC(img, left_rec)
    new_left_rec = get_rectangle_from_mid_bottom(point, rectangle_width, rectangle_height, width)

    point = get_point_from_RANSAC(img, right_rec)
    new_right_rec = get_rectangle_from_mid_bottom(point, rectangle_width, rectangle_height, width)

    draw_rectangle(img, left_rec)
    draw_rectangle(img, right_rec)
    draw_rectangle(img, new_left_rec)
    draw_rectangle(img, new_right_rec)
    show_image(img)


if __name__ == '__main__':
    main()
