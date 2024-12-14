import numpy as np


def leaky_Relu(net: np.ndarray, alpha: float) -> np.ndarray:
    """
    Applies the Leaky ReLU activation function.

    Parameters:
    net (np.ndarray): Input array, can be of any shape.
    alpha (float): Slope for the negative part of the input.

    Returns:
    np.ndarray: Output array after applying the Leaky ReLU function.
    """
    # np.where conditionally applies alpha*net where net <= 0, otherwise applies net
    return np.where(net > 0, net, alpha * net)


def leaky_relu_derivative(net: np.ndarray, alpha: float) -> np.ndarray:
    """
    Computes the derivative of the Leaky ReLU activation function.

    Parameters:
    net (np.ndarray): Input array, can be of any shape.
    alpha (float): Slope for the negative part of the input.

    Returns:
    np.ndarray: Output array of derivatives of the Leaky ReLU function.
    """
    # np.where conditionally returns 1 where net > 0, otherwise returns alpha
    return np.where(net > 0, 1, alpha)