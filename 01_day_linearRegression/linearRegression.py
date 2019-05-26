import numpy as np
import matplotlib.pyplot as plt

m =10
X0 = np.ones((m,1)) # m行 1列
X1 = np.arange(1, m+1).reshape(m, 1) # 生成一个数组 并排列为m行n列矩阵
X = np.mat(np.hstack((X0, X1))) # 矩阵水平拼合（得到截距系数与斜率系数）

Y = np.array([1,2.5,3.2,4.1,5.6,7,7,8,9.6,10.2]).reshape(m, 1) # 数组排列为 m行n列 矩阵
Y = np.mat(Y) # 标准化一下矩阵

alpha = 0.01 # 学习率
finaly_change = 1e-5 # 最小变化幅度 # e是科学计数法的一种表示eN: 10的N次方 负五次方

# 计算代价函数 # 传入θ，实际输入输出
def computeCost(theta, X, Y):
    error = X*theta - Y     # 假设函数 # X是训练集，theta是参数，Y是结果集 ，error是代价函数结果
    J = (1/(2*m))*error.T*error # 返回J 结果 # 平方化error
    return J

# 梯度下降法
def gradientDescent(theta, X, Y):
    error = X*theta -Y      # 假设函数 # X是训练集，theta是参数，Y是结果集 ，error是代价函数结果
    theta_gradient = (1/m)*X.T*error # 返回J 结果 # 平方化error
    return theta_gradient

# 最后就是算法核心部分，梯度下降迭代计算  ！！！！！重要的是 gradient 部分
def gradient_descent(X, Y, alpha):
    # 初始化值，分别为1，1
    theta = np.mat(([1],[1])) # 初始化θ变量
    gradient = gradientDescent(theta, X, Y)  #   ??? 判断第一次最小值；与第一次计算
    while not np.all(np.absolute(gradient)<=finaly_change): # 最小值判断
        theta = theta -alpha*gradient # 第一次gradient 调用第一次的；theta值计算 # theta 上一次theta ；alpha 学习率步长；
        gradient = gradientDescent(theta, X, Y)# 第二次之后再声明一次
    return theta

theta = gradient_descent(X, Y, alpha) # 传参数
print('theta0:', theta[0],'theta1:', theta[1])# 梯度下降结果
print('cost J:', computeCost(theta, X, Y)) # 待见函数结果

# 打印 图像显示
plt.subplots(figsize=(10, 10))  # 声明画布
# X[:,0]是numpy中数组的一种写法，表示对一个二维数组，取该二维数组第一维中的所有数据，
# 第二维中取第0个数据，直观来说，X[:,0]就是取所有行的第0个数据, X[:,1] 就是取所有行的第1个数据
# 画点
X1 = np.mat(X1) # 矩阵化
x1 = list(np.array(X1[:,0])[:,0]) # 取所有行，索引0数据
y1 = list(np.array(Y[:,0])[:,0])
plt.scatter(x1, y1, marker='x')

# 画直线
x2 = list(np.linspace(0, 10, 10))
y2 = []
theta_0 = np.array(theta[0][0])[0][0] # 取数据
theta_1 = np.array(theta[1][0])[0][0]
for i in range(len(x2)):
    y2.append(theta_0 + theta_1*x2[i]) # 绘制每一个条数据
plt.plot(x2, y2, color='red') # 线的颜色

axes = plt.gca() # 声明 一个图
axes.legend(['line regression', 'dataset']) # 显示字段的信息
plt.show() # 展示图层
