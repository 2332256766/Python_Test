import numpy as np
from .sigmoid import sigmoid

def predict(theta, X):
    '''参数计算后，sigmoid，取小数  '''
    m = X.shape[0] # 赋值行列数的行
    return np.round(sigmoid(np.dot(X, theta))) # sigmoid-->X.theta-->获得四舍五入的小数

