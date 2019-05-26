import numpy as np
from .sigmoid import sigmoid

def lrCostFunction(theta, X, Y, _lambda):
    '''与逻辑回归算法相似'''
    m = len(Y)
    theta = theta.reshape(-1, 1)
    _theta = np.copy(theta)
    theta[0, 0] = 0 # ？？？？？？
    s = sigmoid(np.dot(X, theta))

    J = -( np.dot(Y.T, np.log(s))+ np.dot((1-Y).T, np.log(1-s)))
    grad = np.dot(X.T, (s-Y))/m + _lambda*_theta/m
    
    return J[0], grad.reshape(1, -1)[0] # 返回值
