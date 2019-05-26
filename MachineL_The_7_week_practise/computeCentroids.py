import numpy as np

# 计算矩心
def computeCentroids(X, idx, K):
    m = X.shape[0]
    n = X.shape[1]
    centroids = np.zeros((K, n))

    num = np.zeros((K, 1))
    sum_ = np.zeros((K, n))

    for i in range(idx.shape[0]):
        z = idx[i]
        num[z] += 1
        sum_[z,:] += X[i,:]

    centroids = sum_/num
    return centroids