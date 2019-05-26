import numpy as np
from .sigmoid import sigmoid

def costFunctionReg(theta, X, Y, _lambda):
    '''逻辑回归的代价函数部分'''
    m = len(Y)
    theta = theta.reshape(-1,1) # 重新排列theta
    _theta = np.copy(theta) #  拷贝 theta
    _theta[0, 0] = 0 # ???????????
    s = sigmoid(np.dot(X, theta)) # sigmoid 函数 规范数据计算结果
    # J = 函数部分 + 正则化部分
    J = -(np.dot(Y.T, np.log(s)) + np.dot((1-Y).T,np.log(1-s)))/m + _lambda*np.dot(_theta.T, _theta)/m/2
    # 梯度 代价与正则化部分的
    grad = np.dot(X.T, (s-Y))/m + np.dot(_lambda, _theta)/m
    # print(grad)
    # print(grad.reshape(1, -1)[0])
    return J[0], grad.reshape(1, -1)[0] # 取行内的J  取梯度且重新排列
