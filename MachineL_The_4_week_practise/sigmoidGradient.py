import numpy as np
from .sigmoid import sigmoid

def sigmoidGradient(z):
    # sigmoid 的梯度下降
    return sigmoid(z)*(1-sigmoid(z))
