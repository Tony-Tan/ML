import pandas as pds
import numpy as np
import matplotlib.pyplot as plt


class LeastSquaresEstimation():
    def __init__(self, method='OLS'):
        self.method = method

    def fit_line(self, x, y):
        x = np.array(x).reshape(-1, 1)
        # add a column which is all 1s to calculate bias of linear function
        x = np.c_[np.ones(x.size).reshape(-1, 1), x]
        y = np.array(y).reshape(-1, 1)
        if self.method == 'OLS':
            w = np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
            b = w[0][0]
            w = w[1][0]
            return w, b

    def fit_polynomial(self, x, y, d):
        x_org = np.array(x).reshape(-1, 1)
        # add a column which is all 1s to calculate bias of linear function
        x = np.c_[np.ones(x.size).reshape(-1, 1), x_org]
        x_org_d = x_org
        for i in range(1, d):
            x_org_d = x_org_d * x_org
            x = np.c_[x, x_org_d]
        y = np.array(y).reshape(-1, 1)
        w = np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
        return w


def polynomial(w, x, d):
    w = np.array(w).reshape(-1, 1)
    x = np.array(x).reshape(-1, 1)
    x_org_d = x
    X = np.ones(x.size).reshape(-1, 1)
    X = np.c_[X, x_org_d]
    for i in range(1, d):
        x_org_d = x_org_d * x
        X = np.c_[X, x_org_d]
    return X.dot(w)


if __name__ == '__main__':
    data_file = pds.read_csv('./data/1.csv')
    lse = LeastSquaresEstimation()
    x = data_file['x']/320 - 1.
    y = data_file['y']/240 - 1.
    w, b = lse.fit_line(x, y)
    weights_polynomial = lse.fit_polynomial(x, y, 6)
    day_0 = x[0]
    day_end = list(x)[-1]
    days = np.array([day_0,day_end])

    plt.scatter(x, y, c='r', s=30, label='y', marker='o', alpha=0.3)
    plt.scatter(x, polynomial(weights_polynomial, x, 6), c='b', s=30, marker='x', label='polynomial', alpha=0.7)
    plt.plot()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

