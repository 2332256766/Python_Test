import numpy as np
from .sigmoid import sigmoid


def predict(Theta1, Theta2, X):
    '''隐藏层'''
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    a1 = np.vstack((np.ones(m), X.T)).T
    a2 = sigmoid(np.dot(a1, Theta1.T))
    a2 = np.vstack((np.ones(m), a2.T)).T
    a3 = sigmoid(np.dot(a2, Theta2.T))

    return np.argmax(a3, axis=1)
