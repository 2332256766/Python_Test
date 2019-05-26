'''sigmoid 函数'''
# 把计算后的数值 通过函数 缩小到（0~1）区间内
import numpy as np
def sigmoid(z):
    return 1 / ( 1 + np.exp( -z ) )
