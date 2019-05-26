import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


from MachineL_The_2_week_practise.plotData import plotData
from MachineL_The_2_week_practise.mapFeature import mapFeature
from MachineL_The_2_week_practise.costFunctionReg import costFunctionReg
from MachineL_The_2_week_practise.plotDecisionBoundary import plotDecisionBoundary
from MachineL_The_2_week_practise.predict import predict


np.set_printoptions(suppress=True,precision=3)


datas = np.loadtxt('ex2data2.txt', delimiter=',')
X = datas[:, 0:2] ; Y = datas[:,2]
plotData(X, Y)
plt.xlabel('Microship Test 1')
plt.ylabel('Microship Test 2')
# Specified in plot order
plt.legend(['y=1','y=0'])
plt.show()

## =========== Part 1: Regularized Logistic Regression ============
X = mapFeature(X[:,0], X[:,1]).T
print(X.shape) # 发现列数增多
Y = Y.reshape(-1, 1)# 规范化y
# print(Y.shape, y.shape)
# print(type(Y),type(y))
initial_theta = np.zeros(X.shape[1])

_lambda = 1 # 正则化系数

cost, grad = costFunctionReg(initial_theta, X, Y, _lambda) #
print('Cost at initial theta (zeros): %f\n'%cost)

## ============= Part 2: Regularization and Accuracies =============
initial_theta = np.zeros(X.shape[1])
_lambda = 1

# scipy 的方法，最小局部值方法
# 函数， 初始值， method最小化的算法， jac雅各比矩阵  hess黑塞矩阵， 最大迭代次数，参数
res = minimize(costFunctionReg, initial_theta, method='BFGS', jac= True, options={'maxiter':400}, args=(X, Y, _lambda))
# print(res)
theta = res.x.reshape(-1,1) # 取出方法内封装的数据x并排列
print(theta)
print(theta.shape, X.shape, Y.shape)
'''绘图2'''
plotDecisionBoundary(theta, X, Y)
plt.title('lambda = %g'%_lambda)

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')

plt.legend(['y = 1', 'y = 0', 'Decision boundary'])
plt.show()

p = predict(theta, X)
# 成功率 double 小数 乘100倍 %
print('Train Accuracy: %f\n'%(np.mean(np.double(p == Y))*100))

