'''梯度下降法'''

import numpy as np
from .computeCoseMulti import computeCostMulti


def gradientDescentMulti(X, Y, theta, alpha, num_iters):
    m = len(Y)
    J_history = np.zeros((num_iters, 1)) # 创建一个m行1列，零矩阵，记录J
    for iter in range(num_iters):
        error = np.dot(X, theta) - Y
        theta = theta - alpha * np.dot(X.T, error)/m
        J_history[iter][0] = computeCostMulti(X, Y,theta)
    return theta, J_history
