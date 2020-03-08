import numpy as np
import pandas as pds
import matplotlib.pyplot as plt

class Node():
    def __init__(self):
        self.feature = 0
        self.threshold = 0
        self.left_child = None
        self.right_child = None
        self.is_leaf = False
        self.leaf_prediction = 0


def cross_entropy(x):
    x_dict = {}
    x_size = len(x)
    for x_i in x:
        x_i = int(x_i)
        if x_i in x_dict:
            x_dict[x_i] += 1
        else:
            x_dict[x_i] = 1
    entropy = 0
    for x_j in x_dict:
        x_j = int(x_j)
        p = x_dict[x_j]/x_size
        entropy += p * np.log(p+0.0000000001)
    return -entropy


def gini_index(x):
    x_dict = {}
    x_size = len(x)
    for x_i in x:
        x_i = int(x_i)
        if x_i in x_dict:
            x_dict[x_i] += 1
        else:
            x_dict[x_i] = 1
    gini = 0
    for x_j in x_dict:
        x_j = int(x_j)
        p = x_dict[x_j] / x_size
        gini += p * (1 - p)
    return gini


def maximum_element(x):
    x_dict = {}
    for x_i in x:
        x_i = int(x_i)
        if x_i in x_dict:
            x_dict[x_i] += 1
        else:
            x_dict[x_i] = 0
    max_index = -1
    max_value = -1
    for x_i in x_dict:
        if x_dict[x_i] > max_value:
            max_value = x_dict[x_i]
            max_index = x_i
    return max_index


def create_node(x, y, root, depth, max_depth, min_sample_size, method = 'NCE'):
    '''
    :param x:
    :param y:
    :param root:
    :param depth:
    :param max_depth:
    :param min_sample_size:
    :param method: can be 'NCE' and 'GINI' for classification or 'MSE' for regression
    :return:
    '''
    example_size = x.shape[0]
    example_dimension = x.shape[1]
    if example_size <= min_sample_size or depth == max_depth:
        root.is_leaf = True
        if method == 'NCE' or method == 'GINI':
            root.leaf_prediction = maximum_element(y)
        elif method == 'MSE':
            root.leaf_prediction = np.sum(y) / np.size(y)
        root.left_child = None
        root.right_child = None
        return

    measure_matrix = np.zeros(x.shape)
    for i in range(example_dimension):
        for j in range(example_size):
            feature = x[j][i]
            left_y = []
            right_y = []
            for k in range(len(x)):
                if x[k][i] < feature:
                    # left.append(x[k])
                    left_y.append(y[k])
                else:
                    # right.append(x[k])
                    right_y.append(y[k])
            left_y = np.array(left_y)
            right_y = np.array(right_y)
            left_measure = 0
            right_measure = 0
            if np.size(left_y) != 0:
                if method == 'NCE':
                    left_measure = cross_entropy(left_y)
                elif method == 'GINI':
                    left_measure = gini_index(left_y)
                elif method == 'MSE':
                    left_measure = np.sum((left_y - np.sum(left_y) / np.size(left_y)) ** 2)
            if np.size(right_y) != 0:
                if method == 'NCE':
                    right_measure = cross_entropy(right_y)
                elif method == 'GINI':
                    right_measure = gini_index(right_y)
                elif method == 'MSE':
                    right_measure = np.sum((right_y - np.sum(right_y) / np.size(right_y)) ** 2)
            measure_matrix[j][i] = left_measure + right_measure
    # find the minimum position
    min_measure = np.min(measure_matrix)
    min_position = np.where(measure_matrix == min_measure)
    threshold = x[min_position[0][0]][min_position[1][0]]
    feature = min_position[1][0]
    root.threshold = threshold
    root.feature = feature
    # split data into two sets
    left_x = []
    left_y = []
    right_x = []
    right_y = []
    for j in range(example_size):
        if x[j][feature] < threshold:
            left_x.append(x[j])
            left_y.append(y[j])
        else:
            right_x.append(x[j])
            right_y.append(y[j])
    left_x = np.array(left_x)
    left_y = np.array(left_y)
    right_x = np.array(right_x)
    right_y = np.array(right_y)
    if len(left_x) == 0 or len(right_x) == 0 or min_measure ==0:
        root.is_leaf = True
        if method == 'NCE' or method == 'GINI':
            root.leaf_prediction = maximum_element(y)
        elif method == 'MSE':
            root.leaf_prediction = np.sum(y)/np.size(y)
        root.left_child = None
        root.right_child = None
        return
    else:
        root.left_child = Node()
        create_node(left_x, left_y, root.left_child, depth + 1, max_depth, min_sample_size, method)
        root.right_child = Node()
        create_node(right_x, right_y, root.right_child, depth + 1, max_depth, min_sample_size, method)



class CART():
    def __init__(self, max_depth, method='MSE', min_sample_size_in_leaf=2):
        self.max_depth = max_depth
        self.method = method
        self.root = Node()
        self.min_sample_size_in_leaf = min_sample_size_in_leaf

    def training(self, x, y):
        create_node(x, y, self.root, 0, self.max_depth, self.min_sample_size_in_leaf, self.method)

    def test(self, x, node):
        root = node
        while not root.is_leaf:
            if x[root.feature] < root.threshold:
                root = root.left_child
            else:
                root = root.right_child
        return root.leaf_prediction

    def predicting(self, x):
        return self.test(x, self.root)


if __name__ == '__main__':
    data_file = pds.read_csv('./data/4.csv')
    x = np.array([data_file['x_1'] / 320. - 1,data_file['x_2'] / 240. - 1]).transpose()
    y = np.array(data_file['Label']).transpose()
    cart = CART(max_depth=8, method='NCE')
    cart.training(x, y)
    tree = cart.root
    y_hat = np.zeros(y.shape)
    for i in range(y.shape[0]):
        y_hat[i] = cart.predicting(x[i])
    # -----------------------------------visualization---------------------------------------
    plt.figure(1)
    y_dict = {}
    y_hat_dict = {}
    for y_i in y:
        y_i = int(y_i)
        if y_i not in y_dict:
            y_dict[y_i] = np.random.rand(1,3)
    for i in range(0, len(y)):
        plt.scatter(x[i][0], x[i][1], color=y_dict[int(y[i])], marker='o', s=60, alpha=0.5)
    plt.figure(2)
    for i in range(0, len(y)):
        plt.scatter(x[i][0], x[i][1], color=y_dict[int(y_hat[i])], marker='x', s=30, alpha=0.5)
    plt.show()