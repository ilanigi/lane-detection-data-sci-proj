# deprecated
def par_regression_loop(img_path='images/10.jpg', par_height=120, par_width=120, par_amount=3):
    img = general(img_path)

    height, width = img.shape

    x_left = 0
    y_up_left = height - par_height
    x_right = par_width
    y_btm_right = height - 70
    upper_left = x_left, y_up_left
    bottom_right = x_right, y_btm_right
    left_pars = [(upper_left, bottom_right, par_height)]

    x_left = width - 200
    y_up_left = height - 140 - par_height
    x_right = width
    y_btm_right = height
    upper_left = x_left, y_up_left
    bottom_right = x_right, y_btm_right
    right_pars = [(upper_left, bottom_right, par_height)]

    for i in range(par_amount):
        upper_left, bottom_right = right_pars[-1]
        x_left, y_up_left = upper_left

        points = get_data_from_parallelogram(
            img, (upper_left, bottom_right, par_height))
        m, x_on_reg_line, y_on_reg_line = line_params_from_RANSAC(points)

        upper_left, bottom_right = get_par_from_RANSAC_line(
            m, x_on_reg_line, y_on_reg_line, par_width, par_height, x_left)
        right_pars.append((upper_left, bottom_right, par_height))

        upper_left, bottom_right, par_height = left_pars[-1]
        x_left, y_up_left = upper_left

        points = get_data_from_parallelogram(
            img, (upper_left, bottom_right, par_height))
        m, x_on_reg_line, y_on_reg_line = line_params_from_RANSAC(points)
        upper_left, bottom_right = get_par_from_RANSAC_line(m, x_on_reg_line, y_on_reg_line, par_width, par_height,
                                                            x_left)
        left_pars.append((upper_left, bottom_right, par_height))

    for par in right_pars:

        draw_parallelogram(img, par)

    for par in left_pars:
        draw_parallelogram(img, par)