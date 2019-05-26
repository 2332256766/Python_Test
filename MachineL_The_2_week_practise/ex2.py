import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from MachineL_The_2_week_practise.plotData import plotData
from MachineL_The_2_week_practise.costFunction import costFunction
from MachineL_The_2_week_practise.plotDecisionBoundary import plotDecisionBoundary
from MachineL_The_2_week_practise.sigmoid import sigmoid
from MachineL_The_2_week_practise.predict import predict

plt.ion()
np.set_printoptions(suppress=True, precision=2)
print('开始')


data = np.loadtxt('ex2data1.txt', delimiter=',')
# print(data)
X = data[:, 0:2]
Y = data[:, 2]
## ==================== Part 1: Plotting ====================
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')
plotData(X, Y)
plt.xlabel('Exam1_score')
plt.ylabel('Exam2_score')
plt.legend(['Admitted', 'Not admitted'])

# input('\nProgram paused. Press enter to continue.\n')
## ============ Part 2: Compute Cost and Gradient ============

[m,n] = X.shape
X = np.vstack((np.ones(m), X.T)).T
Y = Y.reshape(-1, 1)

theta = np.zeros(n+1)

cost, grad = costFunction(theta, X, Y)

print('Cost at initial theta (zeros): %f\n'%cost)
print('Gradient at initial theta (zeros): \n')
print(grad)
## ============= Part 3: Optimizing using minimize  =============
res = minimize(costFunction, theta, method='BFGS', jac=True, options={'maxiter': 400}, args=(X, Y))

# Print theta to screen
print('Cost at theta found by fminunc: %f\n'%res.fun)
print('theta: \n')
print(res.x)


theta = res.x.reshape(-1, 1)
plotDecisionBoundary(theta, X, Y)

plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Admitted', 'Not admitted'])

input('\nProgram paused. Press enter to continue.\n')
## ============== Part 4: Predict and Accuracies ==============
prod = sigmoid(np.dot([1,45,85], theta))
p = predict(theta, X) # sigmoid方法

# 计算的方法
print('Train Accuracy: %.2f\n'%(np.mean(np.double(p == Y)) * 100))

