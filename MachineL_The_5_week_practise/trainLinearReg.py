import numpy as np
from .linearRegCostFunction import linearRegCostFunction
from scipy.optimize import  minimize


def trainLinearReg(X, Y, _lambda):
    initial_theta = np.zeros(X.shape[1]) # 以列创建theta
    # 计算代价的方法 t -->theta
    costFunction = lambda t:linearRegCostFunction(X, Y, t, _lambda)
    # 寻找最佳梯度值
    res = minimize(costFunction, initial_theta, method='BFGS', jac= True, tol = 1e-2)
    print(res)
    return res.x # 返回x值
