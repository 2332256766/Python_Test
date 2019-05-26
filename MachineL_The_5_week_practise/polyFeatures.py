import numpy as np
from .linearRegCostFunction import linearRegCostFunction
from scipy.optimize import minimize


def polyFeatures(X, p):
    '''特征映射'''
    X_poly = np.zeros((X.shape[0], p))# x行p列零矩阵
    for i in range(p):
        # 第i列赋值
        X_poly[:,i] = np.power(X, i+1).reshape(1, -1)
    print(X_poly)
    return X_poly