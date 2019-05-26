import numpy as np

def pca(X):
    # 特征缩放
    m, n = X.shape
    sigma = np.dot(X.T, X) /m
    U, S, v = np.linalg.svd(sigma, full_matrices=True)
    return U, S
