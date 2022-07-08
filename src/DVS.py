from Utils.Preprocess import general, get_data_from_parallelogram
from src.Utils.Plot import draw_rectangle, show_image, get_rectangle_from_mid_bottom, draw_parallelogram, plot_data
from src.Utils.Regressions import get_point_from_RANSAC, RANSAC


def main():

    img_path = 'images/142.jpg'
    img = general(img_path)

    height, width = img.shape
    par_height = 120
    par_width = 120

    x_left = 0
    y_up_left = height - par_height
    x_right = par_width
    y_btm_right = height - 70
    upper_left = x_left, y_up_left
    bottom_right = x_right, y_btm_right

    # x_left = width - 200
    # y_up_left = height - 140 - length
    # x_right = width
    # y_btm_right = height
    # upper_left = x_left, y_up_left
    # bottom_right = x_right, y_btm_right



    old = (upper_left, bottom_right, par_height)


    draw_parallelogram(img, (upper_left, bottom_right, int(par_height)))
    draw_parallelogram(img, old)
    show_image(img)


if __name__ == '__main__':
    main()
