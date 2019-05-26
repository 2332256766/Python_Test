import numpy as np
from .sigmoid import sigmoid

'''特征值映射'''
# （x1x2 x1^2x2 x1x2^2 ...）
def mapFeature(X1, X2):
    '''x x^2 x^3 。。。'''
    degree = 3 # 幂
    out = np.ones(X1.shape) # 预制X1行列的1矩阵
    for i in range(1, degree+1): # 1 ~ 3
        for j in range(0, i+1): # 第i次 0 ~ 3
            out = np.row_stack((out, np.power(X1, (i-j))* np.power(X2, j))) #  拼行(每次的遍历),运算，x1的幂乘x2的幂
    return out