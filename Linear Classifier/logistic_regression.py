import pandas as pds
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression():

    def logistic_sigmoid(self, a):
        return 1./(1.0 + np.exp(-a))

    def fit(self, x, y, e_threshold, lr=1):
        x = np.array(x)
        # augment the input
        x_dim = x.shape[0]
        x = np.c_[np.ones(x_dim), x]
        # initial parameters in 0.01 to 1
        w = np.random.randint(1, 100, x.shape[1])/100.
        number_of_points = np.size(y)
        for dummy in range(1000):
            y_output = self.logistic_sigmoid(w.dot(x.transpose()))
            # gradient calculation
            e_gradient = np.zeros(x.shape[1])
            for i in range(number_of_points):
                e_gradient += (y_output[i]-y[i])*x[i]
            e_gradient = e_gradient / number_of_points
            # update parameter
            w += -e_gradient*lr
            e = 0
            for i in range(number_of_points):
                e += -(y[i] * np.log(y_output[i]) + (1 - y[i]) * np.log(1 - y_output[i]))
            e /= number_of_points
            if e <= e_threshold:
                break
        return w


if __name__ == '__main__':
    data_file = pds.read_csv('./data/4.csv')
    X = np.array([data_file['x_1'], data_file['x_2']]).transpose()/320.-1
    labels = np.array(data_file['Label'])
    rl = LogisticRegression()
    lr_array = [0.001,0.1,1,2,4,6,8,10,20,40]
    linestyles = ['b-', 'g--', 'b-.', 'c:', 'm-.', 'y-', 'k--', 'c-.', 'g:', 'r:']
    plt.figure(figsize=(12.8,7.2), dpi=100)
    plt.xlim(-10, 1000)
    plt.ylim(-0.1, 1)
    for lr,ls in zip(lr_array,linestyles):
        error = rl.fit(X, labels, 0.0, lr)
        plt.plot(error, ls, label='lr=%2.4f'%lr, alpha=0.7)
    plt.legend(loc='upper right')
    plt.show()