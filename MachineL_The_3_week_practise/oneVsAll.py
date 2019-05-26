import matplotlib.pyplot as plt
import numpy as np

from .lrCostFunction import lrCostFunction
from scipy.optimize import minimize

def oneVsAll(X, Y, num_labels, _lambda):
    ''' 神经网络的梯度下降 '''
    m, n = X.shape # 取行列
    all_theta = np.zeros((num_labels, n+1)) #θ预制
    X = np.vstack((np.ones(m), X.T)).T # x拼行
    Y = Y.reshape(-1, 1) # 重新配列y

    initial_theta = np.zeros(n+1) # 零矩阵
    print('num_labels:\t',num_labels)# 打印数量
    for c in range(num_labels):
        # 求梯度下降最低
        res = minimize(lrCostFunction, initial_theta, method='CG', jac=True, options={'maxiter':50}, args=(X, Y==c, _lambda))
        # 取theta
        all_theta[c] = res.x
    return all_theta
