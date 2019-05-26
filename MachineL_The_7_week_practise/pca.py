import numpy as np

def pca(X):
    m, n = X.shape
    sigma = np.dot(X.T, X)/m
    U, S, v = np.linalg.svd(sigma, full_matrices=True) # 奇异值分解 # M = U\Sigma V^*
    return U, S
