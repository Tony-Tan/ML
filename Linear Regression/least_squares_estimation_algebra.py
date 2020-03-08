import pandas as pds
import numpy as np
import matplotlib.pyplot as plt


data_file = pds.read_csv('./data/babys_weights_by_months.csv')
data_x = np.array(data_file['day'])
data_y_male = np.array(data_file['male'])
data_y_female = np.array(data_file['female'])

data_x_bar = np.mean(data_x)
data_y_male_bar = np.mean(data_y_male)
sum_1 = 0
sum_2 = 0
for i in range(len(data_x)):
    sum_1 += data_x[i]*(data_y_male[i]-data_y_male_bar)
    sum_2 += data_x[i]*(data_x[i]-data_x_bar)
w = sum_1/sum_2
b = data_y_male_bar - w* data_x_bar

day_0 = data_x[0]
day_end = data_x[-1]
days = np.array([day_0,day_end])
plt.plot(days,days*w+b, c='r')
plt.scatter(data_file['day'], data_file['male'], c='r', label='male', alpha=0.5)
#plt.scatter(data_file['day'], data_file['female'], c='b', label='female', alpha=0.5)
plt.xlabel('days')
plt.ylabel('weight(kg)')
plt.legend()
plt.show()

