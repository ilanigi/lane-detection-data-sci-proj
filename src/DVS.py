from Utils.Preprocess import general
from src.Utils.Plot import draw_rectangle, show_image, get_rectangle_from_mid_bottom
from src.Utils.Regressions import get_point_from_RANSAC, RANSAC


def main():
    rectangle_height = 150
    rectangle_width = 250
    img_path = 'images/10.jpg'
    img = general(img_path)

    height, width = img.shape

    left_rec = ((0, height - rectangle_height), (rectangle_width, height - 1))
    right_rec = ((width - rectangle_width, height - rectangle_height), (width - 1, height - 1))

    right_rec_list = [right_rec]
    left_rec_list = [left_rec]

    for i in range(4):
        if i >= 1:
            rectangle_height /= 1.5
            rectangle_width /= 1.5

        left_rec = left_rec_list[-1]
        point = get_point_from_RANSAC(img, left_rec)
        left_rec = get_rectangle_from_mid_bottom(point, rectangle_width, rectangle_height, width)
        left_rec_list.append(left_rec)

        right_rec = right_rec_list[-1]
        point = get_point_from_RANSAC(img, right_rec)
        right_rec = get_rectangle_from_mid_bottom(point, rectangle_width, rectangle_height, width)
        right_rec_list.append(right_rec)

    for rec in right_rec_list:
        draw_rectangle(img, rec)

    for rec in left_rec_list:
        draw_rectangle(img, rec)

    show_image(img)


if __name__ == '__main__':
    main()
