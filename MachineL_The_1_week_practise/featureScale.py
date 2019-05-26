'''特征缩放'''
import numpy as np

np.set_printoptions(suppress=True)
all_data = np.loadtxt(('feature.txt'),delimiter=',')

def featureScale(all_data):
    data =  all_data[:,1:-1]
    X = (data - np.mean(data))/np.std(data)
    Y = all_data[:,-1:]
    X1 = np.ones((len(all_data),1))
    return np.hstack((X1,X,Y))

data = featureScale(all_data)
print(data)
# datas = lambda data : (data - np.mean(data))/np.std(data)
#
# print(datas(data))
