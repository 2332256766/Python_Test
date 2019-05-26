import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.optimize import minimize

from MachineL_The_3_week_practise.displayData import displayData
from MachineL_The_3_week_practise.oneVsAll import oneVsAll
from MachineL_The_3_week_practise.predictOneVsAll import predictOneVsAll

#机器学习在线课堂-练习3第1部分：一对所有
# 指令说明
# ----------------
#此文件包含帮助您开始
#线性运动。您需要完成以下功能
#在本练习中：
#lrcostfunction.py（逻辑回归成本函数）
#一个虚拟球.py
#预测值vsall.py
#预测.py
#对于本练习，您将不需要更改此文件中的任何代码，
#或上述文件以外的任何其他文件。

input_layer_size = 400 #  20x20数字输入图像
num_labels = 10 # 数量标签
## =========== Part 1: Loading and Visualizing Data =============
# 数据的可视化
#我们首先通过加载和可视化数据集来开始练习。
#您将使用包含手写数字的数据集。

# 加载数据集
data = sio.loadmat('ex3data1.mat')
X = data['X']; Y = data['y']%10
print('data.shape:\t',type(data))
m = X.shape[0]

# 随机100 个数据去显示
rand_indices = np.random.permutation(m)
sel = X[rand_indices[0:100]]

displayData(sel)
plt.show()

## ============ Part 2: Vectorize Logistic Regression ============
# 在练习的这一部分中，您将重用逻辑回归
# 上一个练习的代码。你在这里的任务是确保
# 规范化的逻辑回归实现是向量化的。后
# 这样，您将为手写体实现一对所有分类
# 数字数据集。

print('\nTraining One-vs-All Logistic Regression...\n')
# print（'\n一个逻辑回归与所有逻辑回归比较…\n'）

_lambda = 0.1
all_theta = oneVsAll(X, Y, num_labels, _lambda) # 算法

#input（'程序暂停。按Enter继续。\n'）

## ================ Part 3: Predict for One-Vs-All ================
pred = predictOneVsAll(all_theta, X) # 算法
print('\nTraining Set Accuracy: %f\n'%(np.mean(np.double(pred == Y.T)) * 100))

#input（'程序暂停。按Enter继续。\n'）
