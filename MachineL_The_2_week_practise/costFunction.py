import numpy as np
from .sigmoid import sigmoid

def costFunction(theta, X, Y):
    m = len(Y)
    theta = theta.reshape(-1,1) # -1 把theta所有的行列数量相乘的数的行

    print(theta)
    s = sigmoid(np.dot(X, theta)) # sigmoid假设函数
    J = -(np.dot(Y.T,np.log(s))+np.dot((1-Y).T, np.log(1-s)))/m # 逻辑回归J代价函数
    grad = np.dot(X.T, (s-Y))/m # 运算后的X=y ,减去Y # 梯度下降 # ???????? 为什么要梯度下降
    return J[0], grad.reshape(1, -1)[0]
