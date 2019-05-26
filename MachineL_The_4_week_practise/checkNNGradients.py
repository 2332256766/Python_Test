import numpy as np
from .debuglnitializeWeights import debugInitializeWeights
from .computeNumericalGradient import computeNumerucalGradient
from .nnCostFunction import nnCostFunction

def checkNNGradients(_lambda=None):
    if _lambda == None:
        _lambda = 0

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    X = debugInitializeWeights(m, input_layer_size -1)
    Y = ((1+np.arange(m))%num_labels).reshape(-1, 1)

    # Unroll parameters
    nn_params = np.append(Theta1.flatten(), Theta2.flatten())

    # Short hand for cost function
    costFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, Y, _lambda)

    cost, grad = costFunc(nn_params)
    numgrad = computeNumerucalGradient(costFunc, nn_params)

    print(grad)
    print(numgrad)
    print('The above two columns you get should be very similar.\n \
                (Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')

    diff  = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('If your backpropagation implementation is correct, then \n \
          the relative difference will be small (less than 1e-9). \n \
          \nRelative Difference: %g\n'%diff)