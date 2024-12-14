import numpy as np


def elu(net: np.ndarray, alpha: float) -> np.ndarray:
    """
    Applies the Exponential Linear Unit (ELU) activation function.

    Parameters:
    net (np.ndarray): Input array, can be of any shape.
    alpha (float): Scaling factor for the negative part of the input.

    Returns:
    np.ndarray: Output array after applying the ELU function.
    """
    # np.where conditionally applies alpha*(np.exp(net) - 1) where net <= 0, otherwise applies net
    return np.where(net > 0, net, alpha * (np.exp(net) - 1))


def elu_derivative(net: np.ndarray, alpha: float) -> np.ndarray:
    """
    Computes the derivative of the Exponential Linear Unit (ELU) activation function.

    Parameters:
    net (np.ndarray): Input array, can be of any shape.
    alpha (float): Scaling factor for the negative part of the input.

    Returns:
    np.ndarray: Output array of derivatives of the ELU function.
    """
    # np.where conditionally returns 1 where net > 0, otherwise returns np.exp(net) + alpha
    return np.where(net > 0, 1, np.exp(net) + alpha)