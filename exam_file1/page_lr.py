import numpy as np
data1 = np.loadtxt(('pe_train.txt'),delimiter=',')
X_train = data1[:,:-1]
Y_train = data1[:,-1]

data2 = np.loadtxt(('pe_test.txt'),delimiter=',')
X_test = data2[:,:-1]
Y_test = data2[:,-1]

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)

print(model.predict(X_test))
print(Y_test)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(X_test,Y_test)
# plt.show()