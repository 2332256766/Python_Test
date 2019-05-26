from sklearn.neural_network import MLPClassifier
import numpy as np

# np.set_printoptions(suppress=True,precision=-1)
'''加载数据'''
print('加载数据...')
data = np.loadtxt('imagesData.txt',delimiter=',')
data = np.mat(data)
print('行',data.shape[0],'type:',type(data))# 10000行 784列

'''划分集'''
m = data.shape[0]
train_set = data[0:int(m*.6),:]
prov_set = data[int(m*.6):int(m*.8),:]
test_set = data[int(m*.8):,:]

'''训练模型'''# 训练空间 与集类标号 svmlb
print(type(train_set))
t_X = train_set[:,:-1].tolist()
t_Y = train_set[:,-1].tolist()
# 创建模型
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# 传参
clf.fit(t_X, t_Y)
print(type(t_Y),type(t_X))
print('隐藏层数：',clf.n_layers_)
print('迭代次数:',clf.n_iter_)
print('J函数:',clf.loss_)
print('核（激活）函数',clf.out_activation_)

#model = svm_train(t_Y, t_X,'-c 5')
'''函数'''
def ptest(someset):
    print('#'*20)
    import random
    i = random.randint(0, len(someset) - 1)
    X = someset[:, :-1].tolist()[i]
    Y = someset[:, -1].tolist()[i]

    predicted = clf.predict([X])
    print('预测值',predicted[0],'\n实际值：',Y[0])

    predicts = clf.predict_proba([X])# 返回每种标签的预测值
    print('预测概率0',round(predicts[0][0],5),'\n预测概率1',round(predicts[0][1],5))

'''交叉验证集'''
ptest(prov_set)

'''测试集'''
ptest(test_set)
