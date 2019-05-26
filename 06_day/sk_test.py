'''sklearn 的使用'''
import numpy as np

data = np.loadtxt('ex2data2.txt',delimiter=',')# 逻辑回归的数据
X = data[:,:-1]
y = data[:,-1]

def fun1_knn():
    '''数据归一化'''
    # 切割数据集与测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test,Y_train, Y_test = train_test_split(X, y)
    # 归一化处理 X
    from sklearn.preprocessing import StandardScaler
    standardScaler = StandardScaler()
    standardScaler.fit(X_train)
    X_train = standardScaler.transform(X_train)
    X_test = standardScaler.transform(X_test)
    # print(X_test)
    fun2_knn(X_train, X_test, Y_train, Y_test)

def fun2_knn(X_train, X_test,Y_train, Y_test ):
    from sklearn.neighbors import KNeighborsClassifier
    sklearn_knn_clf = KNeighborsClassifier(n_neighbors=6)# 创建一个knn对象
    sklearn_knn_clf.fit(X_train, Y_train)# 用fit创建出模型
    sklearn_knn_clf.score(X_test, Y_test)# 使用训练数据集得出分类准确度
    y_predict = sklearn_knn_clf.predict(X_test)# 使用我们模型预测新的数据
    print(y_predict)

def fun1_LR():
    from sklearn import linear_model
    reg = linear_model.LinearRegression()
    reg.fit([[0,0],[1,1],[2,2]],[0,1,2])
    print(reg.coef_) # array[0.5,0.5]

def fun2_LR():
    from sklearn import linear_model
    r = .5 # 零点5的意思
    # print(r,type(r))
    # c = 8>>4 # 二进制后移>>前移<< 移位
    # c = 2&3
    # c = 1|2|5
    # c = 1^1
    # c = ~1
    c = 2e2
    print(c)

    reg = linear_model.Ridge(alpha= .5)

if __name__ == '__main__':
    # fun1_knn()
    # fun1_LR()
    fun2_LR()
