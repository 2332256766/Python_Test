import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from MachineL_The_5_week_practise.linearRegCostFunction import linearRegCostFunction
from MachineL_The_5_week_practise.trainLinearReg import trainLinearReg
from MachineL_The_5_week_practise.learningCurve import learningCurve
from MachineL_The_5_week_practise.polyFeatures import polyFeatures
from MachineL_The_5_week_practise.featureNormalize import featureNormalize
from MachineL_The_5_week_practise.plotFit import plotFit
from MachineL_The_5_week_practise.validationCurve import validationCurve

print('Loading and Visualizing Data ...\n')

# 加载数据
data = sio.loadmat('ex5data1.mat')
X = data['X'] ; Y = data['y']
Xval = data['X']; Yval = data['yval']
Xtest = data['Xtest'];Ytest = data['ytest']
#
m = X.shape[0]
_X = np.vstack((np.ones(m), X.T)).T
_Xval = np.vstack((np.ones(Xval.shape[0]), Xval.T)).T
_Xtest = np.vstack((np.ones(Xtest.shape[0]), Xtest.T)).T

# 绘制图片
plt.figure()
plt.plot(X, Y, 'rx', ms=5, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

## =========== Part 2: Regularized Linear Regression Cost =============
theta = np.ones(2)
J, _ = linearRegCostFunction(_X, Y, theta, 1)
print('Cost at theta = [1 ; 1]: %f\n(this value should be about 303.993192)\n'%J)

## =========== Part 3: Regularized Linear Regression Gradient =============
theta = np.ones(2)
J, grad = linearRegCostFunction(_X, Y, theta, 1)
print(grad)
print('Gradient at theta = [1 ; 1]:  [%f; %f] ' \
         '\n(this value should be about [-15.303016; 598.250744])\n'%(grad[0], grad[1]))

## =========== Part 4: Train Linear Regression =============
_lambda = 0
theta = trainLinearReg(_X, Y, _lambda) # 测试集线性回归
# 绘图
plt.plot(X, np.dot(_X, theta), '--', linewidth=2)
plt.show()

## =========== Part 5: Learning Curve for Linear Regression =============

_lambda = 0
error_train, error_val = learningCurve(_X, Y, Xval, Yval, _lambda)

plt.figure()
plt.plot(np.arange(m), error_train, np.arange(m), error_val)
plt.title('Learning curve for linear regression')
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.show()
print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('  \t%d\t\t%f\t%f\n'%(i, error_train[i], error_val[i]))

## =========== Part 6: Feature Mapping for Polynomial Regression =============

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)  # Normalize
X_poly = np.vstack((np.ones(m), X_poly.T)).T  # Add Ones

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = np.vstack((np.ones(X_poly_test.shape[0]), X_poly_test.T)).T

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.vstack((np.ones(X_poly_val.shape[0]), X_poly_val.T)).T

print('Normalized Training Example 1:\n')
print(X_poly[0, :])

print('\nProgram paused. Press enter to continue.\n')

## =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of lambda. The code below runs polynomial regression with
#  lambda = 0. You should try running the code with different values of
#  lambda to see how the fit and learning curve change.
#

_lambda = 0
theta = trainLinearReg(X_poly, y, _lambda)

# Plot training data and fit
plt.figure()
plt.plot(X, y, 'rx', ms=5, linewidth=1.5)
plotFit(np.min(X), np.max(X), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial Regression Fit (lambda = %f)' % _lambda)
plt.show()
plt.figure()
error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, _lambda)
plt.plot(np.arange(m), error_train, np.arange(m), error_val)

plt.title('Polynomial Regression Learning Curve (lambda = %f)' % _lambda)
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 100])
plt.legend(['Train', 'Cross Validation'])
plt.show()
print('Polynomial Regression (lambda = %f)\n\n' % _lambda)
print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('  \t%d\t\t%f\t%f\n' % (i, error_train[i], error_val[i]))

# input('Program paused. Press enter to continue.\n')

## =========== Part 8: Validation for Selecting Lambda =============
#  You will now implement validationCurve to test various values of
#  lambda on a validation set. You will then use this to select the
#  "best" lambda value.
#

lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

plt.figure()
plt.plot(lambda_vec, error_train, lambda_vec, error_val)
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('lambda')
plt.ylabel('Error')
plt.show()
print('lambda\t\tTrain Error\tValidation Error\n')
for i in range(lambda_vec.shape[0]):
    print(' %f\t%f\t%f\n' % (lambda_vec[i], error_train[i], error_val[i]))

# input('Program paused. Press enter to continue.\n')