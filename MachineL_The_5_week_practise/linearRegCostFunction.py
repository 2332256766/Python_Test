import numpy as np


def linearRegCostFunction(X, Y, theta,  _lambda):
    m = len(Y) # 行数
    theta = theta.reshape(-1, 1) # theta重新排列
    _theta = np.copy(theta) # 拷贝theta
    _theta[0, 0] =0 # 初始化零
    error = np.dot(X, theta) - Y # 假设函数
    # 代价函数
    J = np.dot(error.T, error)/m/2 + _lambda*np.dot(_theta.T, _theta)/m/2
    # 梯度下降函数
    grad = np.dot(X.T, error)/m + _lambda*_theta/m
    return J, grad.reshape(1, -1)[0]

