import numpy as np
import matplotlib.pyplot as plt

datas = np.loadtxt('./file_data/ex1data1_bak.txt',delimiter=',')
alpha = 0.02
finaly_change = 1e-8
J_history = []

m = (len(datas))
X0 = np.ones((m,1))
X = np.mat(np.hstack((X0,datas[:,0:1])))  # xx行2列 # 截 斜率
Y = np.mat((datas[:,1:2])) # xx行1列

# 代价函数
def CostFunction(X, Y, theta):
    error = np.dot(X, theta) - Y
    return (1/(m*2))*error.T*error # 这个是一个偏导数
# 偏导数 X=1,x
def gradientDescent(X, Y, theta):
    error = np.dot(X, theta) - Y
    return (1/m)*X.T*error # 这个是一个偏导数
# 梯度下降
def gradient_descent(X, Y, alpha):
    theta = np.array(([1],[1])) # 1行2列
    gradient = gradientDescent(X, Y, theta)
    while not np.all(np.absolute(gradient)<=finaly_change):
        theta = theta - alpha*gradient
        gradient = gradientDescent(X, Y, theta)
        J_history.append(CostFunction(X,Y,theta)) # 返回每次的斜率下降数
    return theta

theta = gradient_descent(X, Y, alpha)
print('theta0截距：', theta[0][0],'theta1斜率：', theta[1][0])
print('Cost_J：', CostFunction(X,Y,theta))
iter_num = len(J_history)
print('number_总共有：',iter_num)

'''以下是绘图部分'''
# 梯度下降的图像
x = np.arange(0,iter_num,1) # x轴的位置
y = np.array(J_history).reshape(iter_num,1) #数据的实际值
plt.plot(x,y,'-r') # 绘制
plt.show()

# 线性回归的图像
plt.subplots(figsize=(10,10)) #Figure 绘制窗口大小 # 再打开一个视图
x1 = list(np.array(X[:,1:2])[:,0])
y1 = list(np.array(Y[:,0])[:,0])
plt.scatter(x1, y1,marker='x')

x2 = list(np.linspace(0,np.max(X)+1,10))
y2 = []
theta_0 = np.array(theta[0][0])[0][0]
theta_1 = np.array(theta[1][0])[0][0]
for i in range(len(x2)):
    y2.append(theta_0+theta_1*x2[i])
plt.plot(x2,y2,'-r')

axes = plt.gca()
axes.legend(['line regression','dataset'])
plt.show()
'''测试'''
# begin = 10
# for i in range(begin,begin+5):
#     ipt = datas[i:i+1,0:1]
#     opt = datas[i:i+1,1:2]
#     print('传入x为人数',ipt,'应出y为收入',opt)
#
#     x_predict = float(ipt) #$$ ?????? 输入输出值。。。。。。
#     print(x_predict)
#     y_predict = float(theta[0][0]) + float(theta[1][0])*x_predict
#     print("预测值y为：",y_predict)

# ipt = print(input('输入城市人数x有多少'))
ipt = 1
x_predict = float(ipt)
9
print(x_predict)
y_predict = float(theta[0][0]) + float(theta[1][0])*x_predict
print("预测值收入y为：",y_predict)
