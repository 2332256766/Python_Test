import numpy as np
X = np.mat([[1,1],[1,2],[1,3]])
Y = np.mat([[3],[5],[7]])

def normalEqn(X, Y):
    theta = (X.T*X).I*X.T*X # I逆矩阵
    return theta
result = normalEqn(X, Y)
