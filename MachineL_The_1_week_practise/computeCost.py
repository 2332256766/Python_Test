'''代价函数'''
import numpy as np

def computeCost(X, Y,theta):
    m = 2
    error = np.dot(X, theta) - y
    J = (1/(2*m))*np.dot(error.T, error)
    return J
