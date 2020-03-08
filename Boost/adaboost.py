import numpy as np
import pandas as pds
import matplotlib.pyplot as plt
import numpy as np

# weak classifier
# test each dimension and each value and each direction to find a
# best threshold and direction('<' or '>')
class Stump():

    def __init__(self):
        self.feature = 0
        self.threshold = 0
        self.direction = '<'

    def loss(self,y_hat, y, weights):
        """
        :param y_hat: prediction
        :param y: target
        :param weights:  weight of each data
        :return: loss
        """
        sum = 0
        example_size = y.shape[0]
        for i in range(example_size):
            if y_hat[i] != y[i]:
                sum += weights[i]
        return sum

    def test_in_traing(self, x, feature, threshold, direction='<'):
        """
        test during training
        :param x: input data
        :param feature: classification on which dimension
        :param threshold:  threshold
        :param direction:  '<' or '>' to threshold
        :return: classification result
        """
        example_size = x.shape[0]
        classification_result = -np.ones(example_size)
        for i in range(example_size):
            if direction == '<':
                if x[i][feature] < threshold:
                    classification_result[i] = 1
            else:
                if x[i][feature] > threshold:
                    classification_result[i] = 1
        return classification_result

    def test(self,x):
        """
        test during prediction
        :param x:  input
        :return: classification result
        """
        return self.test_in_traing(x, self.feature, self.threshold, self.direction)

    def training(self, x, y, weights):
        """
        main training process
        :param x: input
        :param y: target
        :param weights: weights
        :return: none
        """
        example_size = x.shape[0]
        example_dimension = x.shape[1]
        loss_matrix_less = np.zeros(np.shape(x))
        loss_matrix_more = np.zeros(np.shape(x))
        for i in range(example_dimension):
            for j in range(example_size):
                results_ji_less = self.test_in_traing(x, i, x[j][i], '<')
                results_ji_more = self.test_in_traing(x, i, x[j][i], '>')
                loss_matrix_less[j][i] = self.loss(results_ji_less, y, weights)
                loss_matrix_more[j][i] = self.loss(results_ji_more, y, weights)
        loss_matrix_less_min = np.min(loss_matrix_less)
        loss_matrix_more_min = np.min(loss_matrix_more)
        if loss_matrix_less_min > loss_matrix_more_min:
            minimum_position = np.where(loss_matrix_more == loss_matrix_more_min)
            self.threshold = x[minimum_position[0][0]][minimum_position[1][0]]
            self.feature = minimum_position[1][0]
            self.direction = '>'
        else:
            minimum_position = np.where(loss_matrix_less == loss_matrix_less_min)
            self.threshold = x[minimum_position[0][0]][minimum_position[1][0]]
            self.feature = minimum_position[1][0]
            self.direction = '<'


class Adaboost():
    def __init__(self, maximum_classifier_size):
        self.max_classifier_size = maximum_classifier_size
        self.classifiers = []
        self.alpha = np.ones(self.max_classifier_size)

    def training(self, x, y, classifier_class):
        """
        training adaboost main steps
        :param x: input
        :param y: target
        :param classifier_class:  what can classifier would be used, here we use stump above
        :return: none
        """
        example_size = x.shape[0]
        weights = np.ones(example_size)/example_size

        for i in range(self.max_classifier_size):
            classifier = classifier_class()
            classifier.training(x, y, weights)
            test_res = classifier.test(x)
            indicator = np.zeros(len(weights))
            for j in range(len(indicator)):
                if test_res[j] != y[j]:
                    indicator[j] = 1

            cost_function = np.sum(weights*indicator)
            epsilon = cost_function/np.sum(weights)
            self.alpha[i] = np.log((1-epsilon)/epsilon)
            self.classifiers.append(classifier)
            weights = weights * np.exp(self.alpha[i]*indicator)

    def predictor(self, x):
        """
        prediction
        :param x: input data
        :return: prediction result
        """
        example_size = x.shape[0]
        results = np.zeros(example_size)
        for i in range(example_size):
            y = np.zeros(self.max_classifier_size)
            for j in range(self.max_classifier_size):
                y[j] = self.classifiers[j].test(x[i].reshape(1,-1))
            results[i] = np.sign(np.sum(self.alpha*y))
        return results


if __name__ == '__main__':
    number_of_classifier = 40
    data_file = pds.read_csv('./data/4.csv')
    X = np.array([data_file['x_1']/ 320. - 1, data_file['x_2']/ 240. - 1]).transpose()
    labels = np.array(data_file['Label'])
    ad = Adaboost(number_of_classifier)
    ad.training(X, labels, Stump)
    # -----------------------------------visualization---------------------------------------
    plt.figure()
    plt.title(str(number_of_classifier)+' classifier(s)')
    for i in range(0, len(labels)):
        if ad.predictor(X[i].reshape(1,-1)) == 1:
            if labels[i] == 1:
                plt.scatter(X[i][0], X[i][1], c='r', marker='o', s=60, alpha=0.5)
            else:
                plt.scatter(X[i][0], X[i][1], c='r', marker='x', s=60, alpha=0.5)
        else:
            if labels[i] == 1:
                plt.scatter(X[i][0], X[i][1], c='b', marker='x', s=60, alpha=0.5)
            else:
                plt.scatter(X[i][0], X[i][1], c='b', marker='o', s=60, alpha=0.5)
    plt.show()