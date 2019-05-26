import numpy as np
import matplotlib.pyplot as plt

# 生成数据集
m = 10
X0 = np.ones((m,1)) # 生成 m行n列
X1 = np.arange(1.5,2.5,0.1).reshape(m,1) # 遍历生成数组并转置m行h列
X = np.hstack((X0,X1))
Y = np.arange(1.6,2.6,0.1).reshape(m,1) # 遍历生成数组并转置m行h列

alpha = 0.01 # 学习率
finaly_change = 1e-5 # 最小变化幅度

# 定义 计算代价函数
def ComputerCost(theta, X, Y):
    Error = np.dot(X, theta)- Y
    return (1/(2*m))*np.dot(Error.T, Error)

# 定义梯度下降函数(偏导数)
def GradientDecent(theta, X, Y):
    Error = np.dot(X, theta)- Y
    return (1/m)*X.T*Error

# 梯度下降迭代计算
def gradient_descent(X, Y, alpha):
    theta = np.mat(([1],[1])) # 初始值
    gradient = GradientDecent(theta, X, Y)
    while not np.all(np.absolute(gradient)<=finaly_change):
        theta = theta - alpha*gradient
        gradient = GradientDecent(theta, X, Y)
    return theta

theta = gradient_descent(X, Y, alpha)
print('theta:', theta)
print('cost J:', ComputerCost(theta, X, Y))
