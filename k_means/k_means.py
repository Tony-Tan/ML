import pandas as pds
import numpy as np
import matplotlib.pyplot as plt


class K_Means():
    """
    input data should be normalized: mean 0, variance 1
    """
    def clusturing(self, x, K):
        """
        :param x: inputs
        :param K: how many groups
        :return: prototype(center of each group), r_nk, which group k does the n th point belongs to
        """
        data_point_dimension = x.shape[1]
        data_point_size = x.shape[0]
        center_matrix = np.zeros((K, data_point_dimension))
        for i in range(len(center_matrix)):
            center_matrix[i] = x[np.random.randint(0, len(x)-1)]

        center_matrix_last_time = np.zeros((K, data_point_dimension))
        cluster_for_each_point = np.zeros(data_point_size, dtype=np.int32)
        # -----------------------------------visualization-----------------------------------
        # the part can be deleted
        #center_color = np.random.randint(0,1000, (K, 3))/1000.
        #plt.scatter(x[:, 0], x[:, 1], color='green', s=30, marker='o', alpha=0.3)
        #for i in range(len(center_matrix)):
        #    plt.scatter(center_matrix[i][0], center_matrix[i][1],  marker='x', s=65, color=center_color[i])
        #plt.show()
        # -----------------------------------------------------------------------------------
        while (center_matrix_last_time-center_matrix).all() != 0:
            # E step
            for i in range(len(x)):
                distance_to_center = np.zeros(K)
                for k in range(K):
                    distance_to_center[k] = (center_matrix[k]-x[i]).dot((center_matrix[k]-x[i]))
                cluster_for_each_point[i] = int(np.argmin(distance_to_center))
            # M step
            number_of_point_in_k = np.zeros(K)
            center_matrix_last_time = center_matrix
            center_matrix = np.zeros((K, data_point_dimension))
            for i in range(len(x)):
                center_matrix[cluster_for_each_point[i]] += x[i]
                number_of_point_in_k[cluster_for_each_point[i]] += 1

            for i in range(len(center_matrix)):
                if number_of_point_in_k[i] != 0:
                    center_matrix[i] /= number_of_point_in_k[i]
            # -----------------------------------visualization-----------------------------------
            # the part can be deleted
            #print(center_matrix)
            #plt.cla()
            #for i in range(len(center_matrix)):
            #    plt.scatter(center_matrix[i][0], center_matrix[i][1], marker='x', s=65,  color=center_color[i])
            #for i in range(len(x)):
            #    plt.scatter(x[i][0], x[i][1], marker='o', s=30, color=center_color[cluster_for_each_point[i]], alpha=0.7)
            #plt.show()
            # -----------------------------------------------------------------------------------
        return center_matrix, cluster_for_each_point


if __name__ == '__main__':
    data_file = pds.read_csv('./data/4.csv')
    X = np.array([(data_file['x_1']-np.mean(data_file['x_1']))/np.std(data_file['x_1']),
                  (data_file['x_2']-np.mean(data_file['x_2']))/np.std(data_file['x_2'])]).transpose()
    km = K_Means()
    c, res = km.clusturing(X, 2)
    print(c)
    print(res)