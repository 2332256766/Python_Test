import numpy as np
from .sigmoid import sigmoid


# 逻辑回归的算法
def costFunctionReg(theta, X, Y, _lambda):
    '''X 运算的值，Y{0,1} theta式子的系数'''
    m = len(Y) # Y 的长度
    theta = theta.reshape(-1, 1) # x行1竖
    _theta = np.copy(theta) # 新建内存 创建内容
    _theta[0, 0] = 0 # 西塔第一个值赋值为0   #???????
    s = sigmoid( np.dot(X, theta)) # sigmoid 函数校正后返回结果
    # 惩罚学习算法
    # 合并后的代价函数J
    J = -(np.dot(Y.T, np.log(s))+np.dot((1-Y).T,np.log(1-s)))/m+_lambda*np.dot(_theta.T, _theta)/m/2
    # 梯度下降的式子
    grad = np.dot(X.T, (s-Y))/m+_lambda*_theta/m
    return J[0], grad.reshape(1, -1)[0]# 返回 代价式子与

