import numpy as np

def featureNormalize(X):
    '''特征缩放'''
    mu = np.mean(X, axis=0) # 均值
    sigma = np.std(X, axis=0) # 标准差
    return (X -mu)/sigma, mu, sigma
