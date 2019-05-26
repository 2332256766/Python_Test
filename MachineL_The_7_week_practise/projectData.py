import numpy as np
#ProjectData仅在项目时计算简化的数据表示形式
#在顶部k特征向量上
#z=项目数据（x，u，k）计算
#归一化输入x到经
#它的前k列返回z中的项目示例

def projectData(X, U, K):
    Z = np.zeros((X.shape[0], K))
    for i in range(X.shape[0]):
        x = X[i, :]
        Z[i,:] = np.dot(x, U[:,0:K])
    return Z
