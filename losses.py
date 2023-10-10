import numpy as np

# Mean squared error loss function and its derivative


def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size


def sse(y_true, y_pred):
    return 0.5 * np.sum(np.power(y_true-y_pred, 2))


def sse_prime(y_true, y_pred):
    return y_pred - y_true


def crossEntropy(y_true, y_pred):
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate cross-entropy loss
    return -np.sum(y_true * np.log(y_pred))


def crossEntropy_prime(y_true, y_pred):
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate the derivative of cross-entropy loss
    return -(y_true / y_pred)
