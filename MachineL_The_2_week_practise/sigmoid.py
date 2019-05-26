import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z)) # 可同时对整个矩阵操作
