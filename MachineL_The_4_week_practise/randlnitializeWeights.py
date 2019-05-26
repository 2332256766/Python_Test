import numpy as np
def randInitializeWeights(L_in, L_out):
    '''初始化权重'''
    epsilon = .12
    return np.random.rand(L_out, L_in+1)* 2* epsilon - epsilon
