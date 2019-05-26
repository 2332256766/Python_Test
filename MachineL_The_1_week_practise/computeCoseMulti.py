'''计算代价函数'''

import numpy as np

def computeCostMulti(X, Y, theta):
    m = len(Y)
    error = np.dot(X, theta) - Y
    J = np.dot(error.T, error)/m/2
    return J

