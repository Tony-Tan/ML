import pandas as pds
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import k_means


test_data_set = 2

def Gaussian( x, u, variance):
    k = len(x)
    return np.power(2*np.pi, -k/2.)*np.power(np.linalg.det(variance),
                                             -1/2)*np.exp(-0.5*(x-u).dot(np.linalg.inv(variance)).dot((x-u).transpose()))


class EM():
    def mixed_Gaussian(self,x,pi,u,covariance):
        res = 0
        for i in range(len(pi)):
            res += pi[i]*Gaussian(x,u[i],covariance[i])
        return res

    def clusturing(self, x, d, initial_method='K_Means'):
        data_dimension = x.shape[1]
        data_size = x.shape[0]
        if initial_method == 'K_Means':
            km = k_means.K_Means()
            # k_means initial mean vector, each row is a mean vector's transpose
            centers, cluster_for_each_point = km.clusturing(x, d)
            # initial latent variable pi

            pi = np.ones(d)/d

            # initial covariance

            covariance = np.zeros((d,data_dimension,data_dimension))
            for i in range(d):
                covariance[i] = np.identity(data_dimension)/10.0
            # calculate responsibility
            responsibility = np.zeros((data_size,d))
            log_likelihood = 0
            log_likelihood_last_time = 0

            # -----------------------------------visualize---------------------------------------
            center_color = np.random.randint(0,1000, (d, 3))/1000.
            ax = plt.gca()
            for i in range(len(centers)):

                plt.scatter(centers[i][0], centers[i][1], marker='x', s=65,  color=center_color[i], label=str(i))
                # draw covariance
                eig_value_0 = np.linalg.eig(covariance[i])[0][0]
                eig_value_1 = np.linalg.eig(covariance[i])[0][1]
                long_axis = np.linalg.eig(covariance[i])[1][1]
                if eig_value_1 > eig_value_0:
                    long_axis = np.linalg.eig(covariance[i])[1][0]
                angle = np.arccos(long_axis.dot([1, 0]) / np.linalg.norm(long_axis))

                ellipse = Ellipse((centers[i][0], centers[i][1]),
                                  np.max([eig_value_0, eig_value_1]) * 10, np.min([eig_value_0, eig_value_1]) * 10
                                  , angle, edgecolor=center_color[i], fc='None', lw=2)
                # draw ellipse
                ax.add_patch(ellipse)
            for i in range(len(x)):
                plt.scatter(x[i][0], x[i][1], marker='o',s=30, color=center_color[cluster_for_each_point[i]],alpha=0.7)
            plt.legend(loc='upper left')
            #plt.xlim([-1.0, 1.0])
            #plt.ylim([-1.0, 1.0])
            plt.savefig('./data/em_sequence/'+str(test_data_set)+'/0.png')
            plt.show()
            # -----------------------------------------------------------------------------------

            for dummy in range(1,1000):

                log_likelihood_last_time = log_likelihood
                # E step:
                # points in each class
                k_class_dict = {i: [] for i in range(d)}
                for i in range(data_size):
                    responsibility_numerator = np.zeros(d)
                    responsibility_denominator = 0
                    for j in range(d):
                        responsibility_numerator[j] = pi[j]*Gaussian(x[i],centers[j],covariance[j])
                        responsibility_denominator += responsibility_numerator[j]
                    for j in range(d):
                        responsibility[i][j] = responsibility_numerator[j]/responsibility_denominator

                # M step:
                N_k = np.zeros(d)
                for j in range(d):
                    for i in range(data_size):
                        N_k[j] += responsibility[i][j]
                for i in range(d):
                    # calculate mean
                    # sum of responsibility multiply x
                    sum_r_x = 0
                    for j in range(data_size):
                        sum_r_x += responsibility[j][i]*x[j]
                    if N_k[i] != 0:
                        centers[i] = 1/N_k[i]*sum_r_x
                    # covariance
                    # sum of responsibility multiply variance
                    sum_r_v = np.zeros((data_dimension,data_dimension))
                    for j in range(data_size):
                        temp = (x[j]-centers[i]).reshape(1,-1)
                        temp_T = (x[j]-centers[i]).reshape(-1,1)
                        sum_r_v += responsibility[j][i]*(temp_T.dot(temp))
                    if N_k[i] != 0:
                        covariance[i] = 1 / N_k[i] * sum_r_v
                    # latent pi
                    pi[i] = N_k[i]/data_size
                # -----------------------------------visualize---------------------------------------
                plt.cla()
                ax = plt.gca()
                for i in range(len(centers)):
                    plt.scatter(centers[i][0], centers[i][1], marker='x', s=65, color=center_color[i],
                                label=str(i))
                    # draw covariance
                    eig_value_0 = np.linalg.eig(covariance[i])[0][0]
                    eig_value_1 = np.linalg.eig(covariance[i])[0][1]
                    # large eigenvalue correspond shorter axis
                    long_axis = np.linalg.eig(covariance[i])[1][0]
                    if eig_value_1 < eig_value_0:
                        long_axis = np.linalg.eig(covariance[i])[1][1]

                    angle = (np.arctan2(long_axis[0], long_axis[1])/(2*np.pi))*360
                    print('angle: '+str(angle))
                    ellipse = Ellipse((centers[i][0], centers[i][1]),
                                      np.max([eig_value_0, eig_value_1])*1.5,
                                      np.min([eig_value_0, eig_value_1])*1.5,
                                      angle, edgecolor=center_color[i], fc='None', lw=2)
                    # draw ellipse
                    ax.add_patch(ellipse)
                for i in range(len(x)):
                    plt.scatter(x[i][0], x[i][1], marker='o', s=30,
                                color=center_color[np.argmax(responsibility[i])],
                                alpha=0.7)
                plt.legend(loc='upper left')
                #plt.xlim([-1.0, 1.0])
                #plt.ylim([-1.0, 1.0])
                plt.savefig('./data/em_sequence/' + str(test_data_set) + '/' + str(dummy) + '.png')
                plt.show()
                # -----------------------------------------------------------------------------------
                log_likelihood = 0
                for i in range(data_size):
                    log_likelihood += np.log(self.mixed_Gaussian(x[i], pi, centers, covariance))

                if np.abs(log_likelihood - log_likelihood_last_time)<0.001:
                    break
                print(log_likelihood_last_time)
        return pi,centers,covariance

if __name__ == '__main__':
    data_file = pds.read_csv('./data/'+str(test_data_set)+'.csv')
    X = np.array([(data_file['x_1'] - np.mean(data_file['x_1'])) / np.std(data_file['x_1']),
                  (data_file['x_2'] - np.mean(data_file['x_2'])) / np.std(data_file['x_2'])]).transpose()
    km = EM()
    pi,c,v=km.clusturing(X, 3)
    print(pi)
    print(c)
    print(v)