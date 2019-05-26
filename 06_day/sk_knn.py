# 自己编写 KNN算法 实现对蓝点的类别预测  0

# 步骤1
# 导包，数据
# 计算测试点到各点的距离
# 步骤2
# 数据放到方法
# 渲染图像


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # 数据模块
from sklearn.neighbors import KNeighborsClassifier # KNN 模块
# 数据量小 用knn高效
X_train = np.array(([1,1],[1,2],[2,1],[2,2],
            [4,4],[4,5],[5,4],[5,5]))
y_train = np.array([0,0,0,1,1,1,1,1])

# x_train,x_test,y_train,y_test = train_test_split(X,y)
# print(x_train,x_test,y_train,y_test)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)



# plt.scatter(X_train[:0], X_train[:,1],color='r')
# plt.scatter(X_train[:0], X_train[:,1],color='b')
# plt.scatter(text_x1[:0], text_x1[:,1],color='y')
# plt.show()

# print(y_train==0) # 返回布尔
# print(X_train[False,0])# 返回空
# print(X_train[True,0])# 返回点
# d = 0
# a=d==1
# print('a',a)

text_x1 = np.mat([3,4])
text_x2 = np.mat([3,3.1])
result1 = knn.predict(text_x1)
result2 = knn.predict(text_x2)
print(result1)
print(result2)

plt.scatter(X_train[y_train==0,0],X_train[y_train==0,1],color='red',marker='o',label='y==0')
plt.scatter(X_train[y_train==1,0],X_train[y_train==1,1],color='blue',marker='x',label='y==1')
plt.legend()
print(text_x1[0,0],text_x1[0,1])
plt.scatter(text_x1[0,0],text_x1[0,1] ,color='y',marker='+',s=100)
plt.scatter(text_x2[0,0],text_x2[0,1] ,color='y',marker='+',s=100)
plt.show()



