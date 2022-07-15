import matplotlib.pyplot as plt
import numpy as np


def find_func(points_l, points_r, im):
    x_l = []
    y_l = []
    x_r = []
    y_r = []
    for p in points_l:
        x_l.append(p[0])
        y_l.append(800 - p[1])
    for p in points_r:
        x_r.append(p[0])
        y_r.append(800 - p[1])
    middle = (min(x_r) + max(x_l)) / 2
    func_l = np.polyfit(x_l, y_l, 2)
    fx_l = np.linspace(0, middle, 800)
    fy_l = np.polyval(func_l, fx_l)
    func_r = np.polyfit(x_r, y_r, 2)
    fx_r = np.linspace(middle, 1280, 800)
    fy_r = np.polyval(func_r, fx_r)
    fig, ax = plt.subplots()
    ax.imshow(im, extent=[0, 1280, 0, 800])
    ax.plot(fx_l, fy_l, '-', color='red')
    ax.plot(fx_r, fy_r, '-', color='green')
    plt.show()

