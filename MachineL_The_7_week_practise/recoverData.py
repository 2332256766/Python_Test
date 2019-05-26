import numpy as np
#recoverdata在使用
# 项目数据
#x_rec=恢复数据（z，u，k）恢复近似值
#已缩减为k维的原始数据。它返回
#x_-rec中的近似重建。
def recoverData(Z, U, K):
    X_rec = np.zeros((Z.shape[0], U.shape[0]))
    for i in range(Z.shape[0]):
        v = Z[i, :]
        for j in range(U.shape[0]):
            X_rec[i, j] = np.dot(v, U[j, 0:K].T)
    return X_rec
