'''正规方程'''
import numpy as np

def normalEqn(X, Y):
    B = np.linalg.pinv(np.dot(X.T, X)) # 预防伪逆矩阵
    theta = np.dot(np.dot(B,X.T), Y) # 正规方程
    return theta

def normalEqn1(X, Y):
    theta = (X.T*X).I*X.T*Y

X = np.mat([[1,1],[1,2],[1,3]])
Y = np.mat([[1+2],[1+4],[1+6]])

print(normalEqn1(X, Y))
