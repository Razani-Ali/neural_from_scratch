import numpy as np


def relu(net: np.ndarray) -> np.ndarray:
    """
    Applies the Rectified Linear Unit (ReLU) activation function.

    Parameters:
    net (np.ndarray): Input array, can be of any shape.

    Returns:
    np.ndarray: Output array after applying the ReLU function.
    """
    # np.maximum selects the element-wise maximum between 0 and net
    return np.maximum(0, net)


def relu_derivative(net: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the Rectified Linear Unit (ReLU) activation function.

    Parameters:
    net (np.ndarray): Input array, can be of any shape.

    Returns:
    np.ndarray: Output array of derivatives of the ReLU function.
    """
    # np.where conditionally returns 1 where net > 0, otherwise returns 0
    return np.where(net > 0, 1, 0)