import numpy as np


def sigmoid(net: np.ndarray) -> np.ndarray:
    """
    Applies the Sigmoid activation function.

    Parameters:
    net (np.ndarray): Input array, can be of any shape.

    Returns:
    np.ndarray: Output array after applying the Sigmoid function.
    """
    # Sigmoid function formula 1 / (1 + exp(-net))
    x = np.clip(-net, -709.0, 709.0)  # To prevent numerical issues
    return 1 / (1 + np.exp(x))


def sigmoid_derivative(net: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the Sigmoid activation function.

    Parameters:
    net (np.ndarray): Input array, can be of any shape.

    Returns:
    np.ndarray: Output array of derivatives of the Sigmoid function.
    """
    # Derivative of sigmoid: sigmoid(net) * (1 - sigmoid(net))
    sig = sigmoid(net)
    return sig * (1 - sig)