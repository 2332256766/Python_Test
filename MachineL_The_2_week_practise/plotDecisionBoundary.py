import matplotlib.pyplot as plt
import numpy as np
from MachineL_The_2_week_practise.plotData import plotData
from MachineL_The_2_week_practise.mapFeature import mapFeature

'''绘图图'''
def plotDecisionBoundary(theta, X, Y):
    plotData(X[:,1:3], Y.T[0]) # X 1 2 列 Y
    print(X[:, 1:3].shape, Y.T.shape)
    if X.shape[1] <= 3: # 列数小于3
        plot_x = np.array([np.min(X[:,1]), np.max(X[:,1])])
        plot_y = (-1/theta[2])*(theta[1]*plot_x+ theta[0])
        plt.plot(plot_x, plot_y) # 绘图
        plt.legend(['Admitted','Not admitted', 'Decision Boundary']) # 字段
        plt.axis([30, 100, 30, 100])
    else:
        # Hera is the grid range
        u = np.linspace(-1, 1.5, 50) # -1~1.5 之间平均50份 生成一行
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((u.shape[0], v.shape[0])) # 生成零矩阵
        # Evaluate z = theta*x over the grid
        for i in range(0, u.shape[0]):
            for j in range(0, v.shape[0]):
                z[i, j] = np.dot(theta.T, mapFeature(u[i], v[j]))
        # !!! important for plot
        u, v = np.meshgrid(u, v)
        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        plt.contour(u, v, z.T, (0,), colors='g', linewidths=2)
