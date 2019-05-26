# 假设函数:h(x)=theta0*x0+theta1*x1+...+theta2*x2=theta*X
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(suppress=True, precision=2)
file_Train = 'mushroomTrain.txt'
file_Text = 'mushroomTest.txt'

input_datas = np.loadtxt(file_Train,delimiter=',')
test_datas = np.loadtxt(file_Text,delimiter=',')
iter_num = 1000      #迭代次数
learningRate=0.005   #学习率
min_change = 1e-5    #梯度变化阈值
m,n = np.shape(input_datas) # 1981 4
theta = np.mat(np.zeros((n))).reshape(n,1)

X = np.mat(np.hstack((np.ones((len(input_datas),1)),input_datas[:,0:-1])))
Y = np.mat(np.reshape(input_datas[:,-1],(len(input_datas),1)))
rest_X = np.mat(np.hstack((np.ones((len(test_datas),1)),test_datas[:,0:-1])))
rest_Y = np.mat(np.reshape(test_datas[:,-1],(len(test_datas),1)))

'''8888888888888888888888888888'''
def printScatter(filename):
    # 读取数据
    data = np.loadtxt(filename, delimiter=',')
    x = data[:, 0:-1]
    y = data[:, -1]
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.scatter(x[pos, 0], x[pos, 1], marker='s', c='r')
    plt.scatter(x[neg, 0], x[neg, 1], marker='x', c='b')
    axes = plt.gca()
    axes.legend(['model_1','model_2'])
    plt.show()
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def computeCost(X, Y, theta):
    m = len(Y)
    z = sigmoid(np.dot(X, theta))
    z[z == 1] = 9e-18
    z[z == 0] = 1e-18 # 结果分类
    J = - (np.dot(Y.T, np.log(z)) + np.dot((1 - Y).T, np.log(1 - z))) / m # 代价函数惩罚结果
    return J

J = computeCost(X, Y, theta)
print("####.J######",J)
def computeDescent(theta, X, Y):
    z = sigmoid(np.dot(X, theta))
    gradient = np.dot(X.T,(z - Y))/m
    return gradient

def gradientDescent(theta,X,Y,iter_num,learningRate):
    i, J_list = 0, []
    gradient = computeDescent(theta,X,Y)
    print('wating a moment...')
    # while not np.all(np.absolute(gradient)<=min_change):
    while (not np.all(np.absolute(gradient)<=min_change)) and (i<iter_num):
        theta = theta - learningRate*gradient
        gradient = computeDescent(theta, X, Y)
        J_list.append(computeCost(X, Y, theta))
        i+=1
        print(i)
    return theta,J_list,iter_num
theta, J_list, iter_num = gradientDescent(theta,X,Y,iter_num,learningRate) # 运算结果
print(theta,iter_num)
'''88888最终结果图绘88888'''
plt.plot(np.arange(0,iter_num,1),np.array(J_list).reshape(iter_num,1),'-r') # 历史迭代的信息图片
plt.show()
# 决策边界,绘图)决策边界printDecisionBoundary()
theta = np.array(theta)
theta0 = theta[0][0]
theta1 = theta[1][0]
theta2 = theta[2][0]
theta3 = theta[3][0]
#打印散点图和拟合直线
'''打印散点图'''
printScatter(file_Train)
plt.figure()
data = np.loadtxt(file_Text, delimiter=',')
mm,nn = np.shape(data) # m行，n列
ones = np.ones((mm, 1))
x = np.hstack((ones,data[:mm,0:-1]))
y = data[:mm, -1].reshape(-1,1)
pos = np.where(y == 1)
neg = np.where(y == 0) # 定位x的行
plt.plot(x[pos, 1], x[pos, 2], marker='x', c='r')
plt.plot(x[neg, 1], x[neg, 2], marker='x', c='b')

# # #拟合直线
x1  = list(np.linspace(0,1,101))
y1 = []
for i in range(101):
    value = np.mat(x[i,:])*np.mat(theta)
    y1.append(round(value[0,0],2))
plt.plot(x1,y1,'-r')
axes = plt.gca()
axes.legend(['model_1','model_2','line'])
plt.show()
# # step9.计算算法的分类准确度
# #取出剩余的训练数据作为输入数据
# predict_Y = sigmoid(np.dot(rest_X,theta))
# predict_Y[predict_Y>=0.5] = 1
# predict_Y[predict_Y<0.5] = 0
# predict_data = predict_Y-rest_Y
# Train_Accuracy = len(predict_data[predict_data==0])/len(rest_Y)
# print('Train Accuracy: %.2f'%( Train_Accuracy* 100)+' %')
