import numpy as np
np.set_printoptions(suppress=True)

data = np.mat(([[123,54236,63643,758,97,34563,614421],
         [442,423,6,17,145,-52,341747],
         [785,58,-234,46,64,34,1248719],
         [15,1435,1345,728,93678,345,131312]
         ]))

X1 = data[:,0:-1]

def featureScale(datas):
    # m,n = data.shape # 四行五列
    avg = np.mean(X1)
    s = np.std(X1)
    data_featual_scale = (X1 - avg)/s
    return data_featual_scale

X1 = featureScale(X1).round(decimals=3)
X0 = np.ones((len(X1),1))

X = np.hstack((X0,X1))
Y = data[:,-1:]

print(X)
