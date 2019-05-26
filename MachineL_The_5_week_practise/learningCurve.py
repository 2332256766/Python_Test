import numpy as np
from .trainLinearReg import trainLinearReg
from .linearRegCostFunction import linearRegCostFunction


def learningCurve(X, Y, Xval, Yval, _lambda):
    m = X.shape[0] # m 行
    error_train = np.zeros(m) # error零矩阵
    error_val = np.zeros(m)
    for i in range(m):
        Xtrain = X[0:i+1] # 0~m的数据
        Ytrain = Y[0:i+1]
        theta = trainLinearReg(Xtrain, Ytrain, _lambda)
        # 代价函数
        error_train[i], _ = linearRegCostFunction(Xtrain,Ytrain,theta,0)
        error_val[i], _ = linearRegCostFunction(Xval,Yval,theta,0)

    return error_train, error_val