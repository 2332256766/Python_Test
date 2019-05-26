import numpy as np
from .polyFeatures import polyFeatures
import matplotlib.pyplot as plt

def plotFit(min_x, max_x, mu, sigma, theta, p):
    x = np.arange(min_x -15, max_x+25, .05).reshape(-1, 1)
    # 特征缩放
    X_ploy = polyFeatures(x, p)# 特征映射
    X_ploy = (X_ploy -mu)/sigma
    # 拼1矩阵
    m = X_ploy.shape[0] # 取行
    X_ploy = np.vstack((np.ones(m), X_ploy.T)).T
    # Plot绘图
    theta = theta.reshape(-1, 1)
    plt.plot(x, np.dot(X_ploy, theta), '--', linewidth=2)
    plt.axis([-70, 70, -50,50])
