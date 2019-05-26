import numpy as np

def featureScale(datas):
    m,n = datas.shape # 四行五列
    data_featual = datas[:,1:n-1]
    avg = np.mean(data_featual)
    s = np.std(data_featual)
    data_featual_scale = (data_featual - avg)/s
    return np.hstack((datas[:,0], data_featual_scale))

datas = np.mat(np.loadtxt('feature.txt', delimiter=','))
print('输入矩阵\n',datas)

data = featureScale(datas)
print('输出矩阵\n',data)
