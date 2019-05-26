import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from MachineL_The_1_week_practise.featureNormalize import FeatureNormalize
from MachineL_The_1_week_practise.normalEqn import normalEqn
from MachineL_The_1_week_practise.gradientDescentMulti import gradientDescentMulti

## ================ Part 1: Feature Normalization ================数据归一
np.set_printoptions(suppress=True)
print('Loading data ...\n')

data = np.loadtxt('ex1data2.txt',delimiter=',')
X, Y = data[:,0:2], data[:,2].reshape(-1,1)
m = len(Y)
# print('First 10 examples from the dataset: \n')
# print(X[0:10,:])
# print(Y[0:10])
# print('Normalizing Features ...\n')
X, mu, sigma = FeatureNormalize(X)
X  = np.vstack((np.ones(m), X.T)).T

## ================ Part 2: Gradient Descent ================梯度下降法
print('Running gradient descent ...\n')
alpha = 0.01
num_iters = 400 # 迭代次数

theta = np.zeros((3,1)) # 初始化  0 , 1都可以 ？？？？但是有什么区别
# theta = np.ones((3,1)) # 初始化
theta, J_history = gradientDescentMulti(X, Y, theta, alpha, num_iters)

plt.plot(np.arange(0, J_history.size,1), J_history, '-g')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()
print(theta,'\n')

temp = np.array([[1.0, 1650.0, 3.0]])
temp[0, 1:3] = (temp[0, 1:3]-mu)/sigma
price = np.dot(temp, theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f\n'%price)

## ================ Part 3: Normal Equations ================正规方程
print('Solving with normal equations...\n')
data = np.loadtxt('ex1data2.txt',delimiter=',')
X, Y = data[:,0:2], data[:, 2].reshape(-1,1)
m = len(Y)
X = np.vstack((np.ones(m), X.T)).T
theta = normalEqn(X, Y)
print('Theta computed from the normal equations: \n')
print(theta, '\n')

temp = np.array([[1.0, 1650.0, 3.0]])
price = np.dot(temp, theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f\n'%price)
