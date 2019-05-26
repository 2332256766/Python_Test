import matplotlib.pyplot as plt
import numpy as np

def plotData(X, Y):
    plt.figure()
    pos = np.where(Y==1); neg = np.where(Y==0) # 是1 否0
    plt.plot(X[pos][:,0], X[pos][:,1], 'k+', linewidth=2, markersize=7)
    plt.plot(X[neg][:,0], X[neg][:,1], 'ko', markerfacecolor= 'y', markersize=7)
