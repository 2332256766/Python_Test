import numpy as np

def sigmoid(z):
    '''sigmoid 函数 数据值缩放入0~1'''
    return 1/(1+np.exp(-z))
