import numpy as np
import torch

EPSILON = 1e-14

def cross_entropy(X, y_1hot, epsilon=EPSILON):
    """Cross Entropy Loss

        Cross Entropy Loss that assumes the input
        X is post-softmax, so this function only
        does negative loglikelihood. EPSILON is applied
        while calculating log.

    Args:
        X: (n_neurons, n_examples). softmax outputs
        y_1hot: (n_classes, n_examples). 1-hot-encoded labels

    Returns:
        a float number of Cross Entropy Loss (averaged)
    """
    X = torch.clamp(X, EPSILON, 1. - EPSILON)

    # 1/sizeof(mini-batch)
    size = X.shape[1]

    # the log of the activation
    log = torch.log(X)

    # the result of multiplying the log of activation and the true label
    mul = torch.mul(y_1hot, log)

    # the result of dividing the result by the negation of the size
    res = torch.div(torch.sum(torch.mul(y_1hot, torch.log(X))), -size)
    return res

def softmax(X):
    """Softmax

        Regular Softmax

    Args:
        X: (n_neurons, n_examples). 

    Returns:
        (n_neurons, n_examples). probabilities
    """
    # n_neurons, n_examples = X.shape
    # print("X: {}".format(X))

    # take the exponent of X
    exp = torch.exp(X)
    # print("exp: {}".format(exp))

    # summing up the exponentiated values in one data for all dimensions
    sm = torch.sum(exp, dim=1, keepdim=True)
    # print("sm: {}".format(sm))

    # dividing the exponent by the sum, getting the percentile
    div = torch.div(exp, sm)
    # print("div: {}".format(div))

    # exponent divided by the sum
    return div

def stable_softmax(X):
    """Softmax

        Numerically stable Softmax

    Args:
               (1, 0)
               (outer_idx, inner_idx)
               (col, row)
        X: (n_neurons, n_examples). 

    Returns:
        (n_neurons, n_examples). probabilities
    """
    # n_neurons, n_examples = X.shape
    # print("X.shape: {}".format(X.shape))
    # print("X: {}".format(X))

    # the maximum value among each example
    maxj = torch.max(X, dim=0, keepdim=True)[0] # [0] takes the actual element
    # print("maxj: {}".format(maxj))

    # balanced X
    bX = X - maxj
    # print("bX: {}".format(bX))

    # take the exponent of X
    exp = torch.exp(bX)
    # print("exp: {}".format(exp))

    # summing up the exponentiated values in one data for all dimensions
    sm = torch.sum(exp, dim=1, keepdim=True)
    # print("sm: {}".format(sm))

    # dividing the exponent by the sum, getting the percentile
    div = torch.div(exp, sm)
    # print("div: {}".format(div))

    # exponent divided by the sum
    return div

def relu(X):
    """Rectified Linear Unit

        Calculate ReLU

    Args:
        X: torch.Tensor

    Returns:
        A torch.Tenor whereThe shape is the same as X but clamped on 0
    """
    return torch.maximum(X, torch.tensor(0))


def sigmoid(X):
    """Sigmoid Function

        Calculate Sigmoid

    Args:
        X: torch.Tensor

    Returns:
        A torch.Tensor where each element is the sigmoid of the X.
    """
    X_nexp = torch.exp(-X)
    return 1.0 / (1 + X_nexp)
