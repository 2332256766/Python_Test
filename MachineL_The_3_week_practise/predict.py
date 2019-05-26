import numpy as np
from .sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    '''隐藏层'''
    '''θ1，θ2  多个sigmoid 连续计算'''
    m = X.shape[0] # 行数
    num_labels = Theta2.shape[0] # 行数

    a1 = np.vstack((np.ones(m), X.T)).T #
    z2 = np.dot(a1, Theta1.T) #
    a2 = sigmoid(z2) # sigmoid

    a2 = np.vstack((np.ones(m), a2.T)).T
    z3 = np.vstack((np.ones(m), a2.T)).T
    a3 = sigmoid(z3)
    return np.argmax(a3, axis=1) # 取行最大值的索引