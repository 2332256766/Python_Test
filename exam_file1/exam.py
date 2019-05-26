import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, precision=2)
input_datas = np.loadtxt(('pe_train.txt'),delimiter=',')
m,n = np.shape(input_datas) # 2881 5
def featureScale(datas):
    data_featual = datas[:,0:-1]
    avg = np.mean(data_featual)
    s = np.std(data_featual)
    data_featual_scale = (data_featual - avg)/s
    X_0 = np.ones((len(data_featual),1))
    result = np.hstack((X_0, data_featual_scale))
    return result

X_datas = np.mat(featureScale(input_datas))
Y_datas = np.mat(np.reshape(input_datas[:,-1],(len(input_datas),1)))
finaly_change = 1e1
alpha = 0.001
J_history = []

# 定义 计算代价函数
def ComputerCost(theta, X, Y):
    Error = np.dot(X, theta)- Y
    return (1/(2*m))*np.dot(Error.T, Error)

# 定义梯度下降函数(偏导数)
def GradientDecent(theta, X, Y):
    Error = np.dot(X, theta)- Y
    return (1/m)*X.T*Error

# 梯度下降迭代计算
def gradient_descent(X, Y, alpha):
    theta = np.mat((np.ones((n,1)))) # 初始值
    gradient = GradientDecent(theta, X, Y)
    while not np.all(np.absolute(gradient)<=finaly_change):
        theta = theta - alpha*gradient
        gradient = GradientDecent(theta, X, Y)
        J_history.append(ComputerCost(theta,X,Y))
    return theta

theta = gradient_descent(X_datas, Y_datas, alpha)
print('over:',theta)
print('Cost_J：', ComputerCost(theta,X_datas,Y_datas))

iter_num = len(J_history)
print('number_总共有：',iter_num)
print(J_history)

'''以下是绘图部分'''
# 梯度下降的图像
x = np.arange(0,iter_num,1) # x轴的位置
y = np.array(J_history).reshape(iter_num,1) #数据的实际值
plt.plot(x,y,'-r') # 绘制
plt.show()
'''##########'''
# 线性回归的图像
plt.subplots(figsize=(10,10)) #Figure 绘制窗口大小 # 再打开一个视图
# x1 = np.array(X_datas[:,:-1])
# y1 = np.array(Y_datas[:,-1])
# plt.scatter(x1, y1,marker='x')
x2 = list(np.linspace(0,np.max(X_datas)+1,10))
y2 = []
theta_0 = np.array(theta[0][0])[0][0]
theta_1 = np.array(theta[1][0])[0][0]
for i in range(len(x2)):
    y2.append(theta_0+theta_1*x2[i])
plt.scatter(x2,y2,'-r')
axes = plt.gca()
axes.legend(['line regression','dataset'])# 真实值 预测值
plt.show()


'''测试'''
X = X_datas[0:1,:]
print(X_datas,Y_datas[0:1,:])
y_predict = X*theta
print("预测值y为：",y_predict)
