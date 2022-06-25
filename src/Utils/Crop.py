def linear_equation(base_point, mid_point):
    base_x, base_y = base_point
    x_middle_point, y_middle_point, = mid_point
    m = (y_middle_point - base_y) / (x_middle_point - base_x)
    n = - (m * x_middle_point - y_middle_point)
    return lambda x: m * x + n


def set_triangle_scope(img, mid_point=(640, 200), base_height=650):
    height, width = img.shape
    left_linear_equation = linear_equation((0, base_height), mid_point)
    right_linear_equation = linear_equation((width, base_height), mid_point)
    for i in range(height):
        for j in range(width):
            if left_linear_equation(j) > i or right_linear_equation(j) > i:
                img[i, j] = 0
    return img


def get_bottom_right_corner(img):
    height, width = img.shape
    return img[int(height / 2) - 50:, int(width / 2): ]


def crop_rectangle(img,rectangle):
    upper_left, bottom_right = rectangle
    x_left, y_left = upper_left
    x_right, y_right = bottom_right
    img = img[y_left:y_right, x_left:x_right]
    return img
