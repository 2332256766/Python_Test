import matplotlib.pyplot as plt
import numpy as np

def plotData(x, y):
    plt.plot(x,y,'x') # z最后参数是模式选择
    plt.show()

listx = [1,2,3]
listy = [1,2,3]
# plotData(listx, listy)

data = np.loadtxt('ex1data1_bak.txt', delimiter=',')
# print(data)
# print(type(data))
# print(data.shape) # 显示行列数量

X = data[:,0:1] # 第一列
Y = data[:,1:2] # 第二列

listx = np.array(X)
listy = np.array(Y)
print(type(listx))
# print(Y)
plotData(listx, listy)
