from Utils.Regressions import main_par_regression_loop
from function import find_func

def main():
    left_points, right_points = main_par_regression_loop()
    find_func(left_points, right_points, 'images/10.jpg')
    
if __name__ == '__main__':
    main()
