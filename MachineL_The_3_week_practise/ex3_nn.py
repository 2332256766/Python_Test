import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.optimize import minimize

from MachineL_The_3_week_practise.displayData import displayData
from MachineL_The_3_week_practise.oneVsAll import oneVsAll
from MachineL_The_3_week_practise.predict import predict

'''预制信息'''
input_layer_size = 400 # 输入 层 大小 # 20x20数字输入图像
hidden_layer = 25 # # 隐藏？？？层 # 25 hidden units
num_labels =10 # 10行 # 标签数（请注意 我们已将’0‘映射到标签10）

## =========== Part 1: Loading and Visualizing Data =============
# 数据加载与可视化
#我们首先通过加载和可视化数据集来开始练习。
#您将使用包含手写数字的数据集。

data = sio.loadmat('ex3data1.mat') # 数据集提取
X = data['X']; y =data['y']%10 # 取数据 10列
m = X.shape[0] # 取行

print(m) # 5000行

rand_indices = np.random.permutation(m) # 随机数据集
sel = X[rand_indices[0:100]] #

displayData(sel)# 渲染
plt.show()
## ================ Part 2: Loading Pameters ================
# 加载子表
#在练习的这一部分中，我们加载一些预初始化的
#神经网络参数。
print('\nLoading Saved Neural Network Parameters ...\n') # 加载保存的神经网络参数
# 加载theta
data = sio.loadmat('ex3weights.mat')
Theta1 = data['Theta1']; Theta2 = data['Theta2']
print("Theta1.shape",Theta1.shape)
print("Theta2.shape",Theta2.shape)

## ================= Part 3: Implement Predict =================
# 在训练神经网络之后，我们想用它来预测
# 标签。你现在将实现“预测”函数的使用
# 神经网络预测训练集的标签。这封信
# 您计算了培训设置的精度。
pred = (predict(Theta1, Theta2, X)+1)%10

print('\nTraining Set Accuracy: %f\n'%(np.mean(np.double(pred == y.T)) * 100))
input("pause")

#为了让您了解网络的输出，您还可以运行
#通过这些例子一次一个，看看它在预测什么。

rp = np.random.permutation(m)# m是行数
print(rp)
for i in range(m):
    # display
    print('\nDisplating Example Image\n')
    t = np.array([X[rp[i]]]) # ？？？

    displayData(t)
    pred = predict(Theta1, Theta2, t)
    print('\nNeural Network Prediction: %d (digit %d)\n'%(pred, (pred+1)%10))
    # input('Program paused. Press enter to continue.\n')

print('over')