import numpy as np
from .sigmoid import sigmoid
from .sigmoidGradient import sigmoidGradient

#nNCostFunction实现两层的神经网络成本函数
#进行分类的神经网络
#[J Grad]=nncostFuncton（nn_参数，隐藏层大小，num_标签，……
#x，y，lambda）计算神经网络的成本和梯度。这个
#神经网络的参数被“展开”到向量中。
#nn_参数，需要转换回权重矩阵。
#返回的参数梯度应为
#神经网络的偏导数。

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, _lambda):
    Theta1 = nn_params[0:hidden_layer_size * (input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = nn_params[hidden_layer_size * (input_layer_size+1):].reshape(num_labels, hidden_layer_size + 1)
    m = len(Y)
    # ===============此处是您的代码=========
    # 说明：您应该通过
    # 以下部分。
    # 第1部分：前馈神经网络并返回
    # 变量j.在实现第1部分后，您可以验证
    # 通过验证成本，成本函数计算是正确的。
    # 按ex4.m计算
    # 第2部分：实现反向传播算法计算梯度
    # Theta1_Grad和Theta2_Grad。你应该返回的偏导数
    # θ1和θ2的成本函数
    # θ2_梯度。在执行第2部分之后，您可以检查
    # 通过运行checknnGradients，您的实现是正确的
    # 注：传递给函数的向量y是标签向量
    # 包含1..k的值。您需要将此向量映射到
    # 与神经网络一起使用的1和0的二进制矢量
    # 成本函数。
    # 提示：我们建议使用for循环实现backpropagation
    # 如果您正在为
    # 第一次。
    # 第三部分：利用成本函数和梯度实现正则化。
    # 提示：您可以围绕
    # 反向传播。也就是说，可以计算
    # 分别进行正则化，然后将它们添加到theta1_Grad中。
    # 以及第2部分的θ_梯度。
    a1 = np.vstack((np.ones(m), X.T)).T
    a2 = sigmoid(np.dot(a1, Theta1.T))
    a2 = np.vstack((np.ones(m), a2.T)).T
    a3 = sigmoid(np.dot(a2, Theta2.T))
    # tile????? 瓷砖 -->把数组像瓷砖一样铺展开（矩阵，（m行，n列））
    Y = np.tile((np.arange(num_labels)+1)%10,(m,1))==np.tile(Y,(1, num_labels))

    print("*"*50)
    print(Y.shape)
    print(Y)

    regTheta1 = Theta1[:,1:]
    regTheta2 = Theta2[:,1:]

    J = -np.sum(Y*np.log(a3) + (1-Y)*np.log(1-a3))/m + _lambda*np.sum(regTheta1*regTheta1)/m/2 +_lambda*np.sum(regTheta2*regTheta2)/m/2

    delta1 = np.zeros(Theta1.shape)
    delta2 = np.zeros(Theta2.shape)
    for i in range(m):
        a1_ = a1[i];    a2_ =a2[i];     a3_ = a3[i]
        d3 = a3_ - Y[i]
        # append 变成行
        d2 = np.dot(d3,Theta2)*sigmoidGradient(np.append(1,np.dot(a1_, Theta1.T)))
        delta1 = delta1 + np.dot(d2[1:].reshape(-1,1),a1_.reshape(1,-1))
        delta2 = delta2 + np.dot(d3.reshape(-1, 1), a2_.reshape(1, -1))

    regTheta1 = np.vstack((np.zeros(Theta1.shape[0]), regTheta1.T)).T
    regTheta2 = np.vstack((np.zeros(Theta2.shape[0]), regTheta2.T)).T
    Theta1_grad = delta1/m + _lambda *regTheta1/m
    Theta2_grad = delta2/m + _lambda *regTheta2/m

    # flatten--> 折叠为一维数组
    grad = np.append(Theta1_grad.flatten(), Theta2_grad.flatten())
    print('cost value: %lf'%J)
    return J, grad


