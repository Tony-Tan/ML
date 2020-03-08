import pandas as pds
import numpy as np
import matplotlib.pyplot as plt
import cv2

class LinearClassifier():
    def least_square(self, x, y):
        x = np.array(x)
        x_dim = x.shape[0]
        x = np.c_[np.ones(x_dim), x]
        w = np.linalg.pinv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
        return w.transpose()

    def fisher(self, x, y):
        x = np.array(x)
        x_dim = x.shape[1]
        m_1 = np.zeros(x_dim)
        m_1_size = 0
        m_2 = np.zeros(x_dim)
        m_2_size = 0
        for i in range(len(y)):
            if y[i] == 0:
                m_1 = m_1 + x[i]
                m_1_size += 1
            else:
                m_2 = m_2 + x[i]
                m_2_size += 1
        if m_1_size != 0 and m_2_size != 0:
            m_1 = (m_1/m_1_size).reshape(-1, 1)
            m_2 = (m_2/m_2_size).reshape(-1, 1)
        s_c_1 = np.zeros([x_dim, x_dim])
        s_c_2 = np.zeros([x_dim, x_dim])
        for i in range(len(y)):
            if y[i] == 0:
                s_c_1 += (x[i] - m_1).dot((x[i] - m_1).transpose())
            else:
                s_c_2 += (x[i] - m_2).dot((x[i] - m_2).transpose())
        s_w = s_c_1 + s_c_2
        return np.linalg.inv(s_w).dot(m_2-m_1)


def label_convert(y, method ='1-of-K'):
    if method == '1-of-K':
        label_dict = {}
        number_of_label = 0
        for i in y:
            if i not in label_dict:
                label_dict[i] = number_of_label
                number_of_label += 1
        y_ = np.zeros([len(y),number_of_label])
        for i in range(len(y)):
            y_[i][label_dict[y[i]]] = 1
        return y_,number_of_label


if __name__ == '__main__':
    for data_number in range(1,9,1):
        data_file = pds.read_csv('./data/'+str(data_number)+'.csv')
        X = np.array([data_file['x_1'],data_file['x_2']]).transpose()
        labels = np.array(data_file['Label'])

        '''
        Y, number_of_class = label_convert(labels, method='1-of-K')
        lc = LinearClassifier()
        w = lc.least_square(X, Y)
        print(w)
        width = 640
        height = 480
        image = np.ones([height, width, 3], dtype=np.uint8)
        for j in range(width):
            for i in range(height):
                x = np.array([1, j,i])
                y_ = w.dot(x)
                value = np.argmax(y_)+1
                image[height-1-i, j] = [int(200*(value/float(number_of_class))),
                                   int(200*(value/float(number_of_class))),
                                   int(200*(value/float(number_of_class)))]
        label_dict = {}
        for i in range(len(labels)):
            if labels[i] not in label_dict:
                rgb = np.random.randint(0,255,3)
                label_dict[labels[i]] = (int(rgb[0]),int(rgb[1]),int(rgb[2]))
            cv2.circle(image, (X[i][0], height - X[i][1]), 2, label_dict[labels[i]], thickness=3)

        cv2.imshow('Image', image)
        cv2.imwrite('./data/'+str(data_number)+'.png',image)

        '''
        lc = LinearClassifier()
        w = lc.fisher(X, labels)
        w = w/np.linalg.norm(w)
        for i in range(len(labels)):
            if labels[i] == 1:
                plt.scatter(X[i][0], X[i][1], c='r', alpha=0.5)
            else:
                plt.scatter(X[i][0], X[i][1], c='b', alpha=0.5)
        plt.plot([320,w[0]*100],[0,w[1]*100])
        plt.show()
