'''·正规方程·'''
import numpy as np


def FeatureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X-mu)/sigma, mu, sigma
# A = np.mat([[1,4],[2,3]])
# print(featureNormalizes(A))
