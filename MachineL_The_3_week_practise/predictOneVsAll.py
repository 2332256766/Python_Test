import numpy as np
from .sigmoid import sigmoid
# 取重要特征

def predictOneVsAll(all_theta, X):
    '''sigmoid 函数 '''
    m = X.shape[0] # 取列
    X = np.vstack((np.ones(m), X.T)).T # 拼X=1
    # 返回列方向上最大数值下标
    return np.argmax(sigmoid(np.dot(all_theta, X.T)), axis=0) # axis 0 代表列方向 1代表行方向
