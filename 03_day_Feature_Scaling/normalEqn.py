'''正规方程'''

import numpy as np

X = np.mat([[1,1],[1,2],[1,3]])
Y = np.mat(([1+2],[1+4],[1+6]))

def normalEqn(X, Y):
    # X^2
    theta = (X.T*X).I*X.T*Y
    return theta

theta = normalEqn(X, Y)
print(theta)