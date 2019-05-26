import numpy as np

def sigmoid(z):
    '''作用：结果缩放'''
    return 1/(1+ np.exp(-z))
