import numpy as np
from .trainLinearReg import trainLinearReg
from .linearRegCostFunction import linearRegCostFunction


def validationCurve(X, Y, Xval, Yval):
    # 初始化lambda 值
    lambda_vec = np.array([0,.001,.003,.01,.03,.1,.3,.1,1,3,10])
    m = lambda_vec.shape[0]
    # 初始化矩阵
    error_train = np.zeros(m)
    error_val = np.zeros(m)
    for i in range(m):
        _lambda = lambda_vec[i]# 遍历lambda赋值
        theta = trainLinearReg(X, Y, _lambda) # 线性回归
        error_train[i], _ = linearRegCostFunction(X, Y, theta, 0) # 训练集代价函数
        error_val[i], _ = linearRegCostFunction(Xval, Yval, theta, 0) # 测试集代价函数
    return lambda_vec, error_train, error_val
