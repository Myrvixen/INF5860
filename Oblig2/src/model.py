#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                                                                               #
# Part of mandatory assignment 2 in                                             #
# INF5860 - Machine Learning for Image analysis                                 #
# University of Oslo                                                            #
#                                                                               #
#                                                                               #
# Ole-Johan Skrede    olejohas at ifi dot uio dot no                            #
# 2018.03.01                                                                    #
#                                                                               #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

"""Define the dense neural network model"""

import numpy as np
from scipy.stats import truncnorm


def one_hot(Y, num_classes):
    """Perform one-hot encoding on input Y.

    It is assumed that Y is a 1D numpy array of length m_b (batch_size) with integer values in
    range [0, num_classes-1]. The encoded matrix Y_tilde will be a [num_classes, m_b] shaped matrix
    with values

                   | 1,  if Y[i] = j
    Y_tilde[i,j] = |
                   | 0,  else
    """
    m = len(Y)
    Y_tilde = np.zeros((num_classes, m))
    Y_tilde[Y, np.arange(m)] = 1
    return Y_tilde


def initialization(conf):
    """Initialize the parameters of the network.

    Args:
        layer_dimensions: A list of length L+1 with the number of nodes in each layer, including
                          the input layer, all hidden layers, and the output layer.
    Returns:
        params: A dictionary with initialized parameters for all parameters (weights and biases) in
                the network.
    """
    # TODO: Task 1

    layer_dims = conf['layer_dimensions']
    L = len(layer_dims)

    params = {}

    for i in range(1, L):

        W = np.random.normal(0, np.sqrt(2/layer_dims[i-1]), size=(layer_dims[i-1], layer_dims[i]))

        b = np.zeros((1, layer_dims[i]))

        params['W'+ str(i)] = W
        params['b'+str(i)] = b

    return params


def activation(Z, activation_function):
    """Compute a non-linear activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 2 a)
    if activation_function == 'relu':
        return Z * (Z >= 0)
    else:
        print("Error: Unimplemented derivative of activation function: {}", activation_function)
        return None


def softmax(Z):
    """Compute and return the softmax of the input.

    To improve numerical stability, we do the following

    1: Subtract Z from max(Z) in the exponentials
    2: Take the logarithm of the whole softmax, and then take the exponential of that in the end

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 2 b)

    # Stability trick 1
    Z -= np.max(Z)

    # Stability trick 2
    S = Z - np.log(np.sum(np.exp(Z), axis=0))
    S = np.exp(S)

    return S


def forward(conf, X_batch, params, is_training):
    """One forward step.

    Args:
        conf: Configuration dictionary.
        X_batch: float numpy array with shape [n^[0], batch_size]. Input image batch.
        params: python dict with weight and bias parameters for each layer.
        is_training: Boolean to indicate if we are training or not. This function can namely be
                     used for inference only, in which case we do not need to store the features
                     values.

    Returns:
        Y_proposed: float numpy array with shape [n^[L], batch_size]. The output predictions of the
                    network, where n^[L] is the number of prediction classes. For each input i in
                    the batch, Y_proposed[c, i] gives the probability that input i belongs to class
                    c.
        features: Dictionary with
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
               We cache them in order to use them when computing gradients in the backpropagation.
    """
    # TODO: Task 2 c)

    layer_dims = conf['layer_dimensions']
    L = len(layer_dims)
    act_func = conf['activation_function']

    features = {}
    features['A_0'] = X_batch

    """ Cave-man programming
    Z_1 = np.dot(params['W_1'].T, X_batch) + params['b_1']
    features['Z_1'] = Z_1.round(2)

    A_1 = activation(Z_1, conf['activation_function'])
    features['A_1'] = A_1

    Z_2 = np.dot(params['W_2'].T, A_1) + params['b_2']
    features['Z_2'] = Z_2.round(2)

    Y_proposed = softmax(Z_2)
    """

    for i in range(1,L):

        W = params['W_'+ str(i)]
        b = params['b_' + str(i)]
        A = features['A_' + str(i-1)]

        Z = np.dot(W.T, A) + b
        features['Z_' + str(i)] = Z.round(2)

        if i != (L-1):
            A_next = activation(Z, act_func)
            features['A_'+ str(i)] = A_next
        else:
            A_next = softmax(Z)
            features['A_' + str(i)] = A_next
            Y_proposed = A_next


    return Y_proposed, features


def cross_entropy_cost(Y_proposed, Y_reference):
    """Compute the cross entropy cost function.

    Args:
        Y_proposed: numpy array of floats with shape [n_y, m].
        Y_reference: numpy array of floats with shape [n_y, m]. Collection of one-hot encoded
                     true input labels

    Returns:
        cost: Scalar float: 1/m * sum_i^m sum_j^n y_reference_ij log y_proposed_ij
        num_correct: Scalar integer
    """
    # TODO: Task 3

    m = Y_proposed.shape[1]

    cost = -1.0/m*np.sum(Y_reference*np.log(Y_proposed))
    num_correct = np.sum(np.argmax(Y_proposed, axis=0) == np.argmax(Y_reference, axis=0))

    return cost, num_correct


def activation_derivative(Z, activation_function):
    """Compute the gradient of the activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 4 a)
    if activation_function == 'relu':
        return 1*(Z >= 0)
    else:
        print("Error: Unimplemented derivative of activation function: {}", activation_function)
        return None


def backward(conf, Y_proposed, Y_reference, params, features):
    """Update parameters using backpropagation algorithm.

    Args:
        conf: Configuration dictionary.
        Y_proposed: numpy array of floats with shape [n_y, m].
        features: Dictionary with matrices from the forward propagation. Contains
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
        params: Dictionary with values of the trainable parameters.
                - the weights W^[l] for l in [1, L].
                - the biases b^[l] for l in [1, L].
    Returns:
        grad_params: Dictionary with matrices that is to be used in the parameter update. Contains
                - the gradient of the weights, grad_W^[l] for l in [1, L].
                - the gradient of the biases grad_b^[l] for l in [1, L].
    """
    # TODO: Task 4 b)

    grad_params = {}

    m = Y_proposed.shape[1]

    """ Cave-man programming
    J_2 = Y_proposed - Y_reference

    grad_b_2 = 1/m*np.dot(J_2,np.ones((m,1)))

    grad_W_2 = 1/m*np.dot(features['A_1'], J_2.T)

    J_1 = activation_derivative(features['Z_1'], 'relu')*np.dot(params['W_2'], J_2)

    grad_b_1 = 1/m*np.dot(J_1, np.ones((m,1)))

    grad_W_1 = 1/m*np.dot(features['A_0'], J_1.T)


    grad_params['grad_W_1'] = grad_W_1
    grad_params['grad_W_2'] = grad_W_2
    grad_params['grad_b_1'] = grad_b_1
    grad_params['grad_b_2'] = grad_b_2
    """

    return grad_params


def gradient_descent_update(conf, params, grad_params):
    """Update the parameters in params according to the gradient descent update routine.

    Args:
        conf: Configuration dictionary
        params: Parameter dictionary with W and b for all layers
        grad_params: Parameter dictionary with b gradients, and W gradients for all
                     layers.
    Returns:
        updated_params: Updated parameter dictionary.
    """
    # TODO: Task 5
    updated_params = None
    return updated_params
