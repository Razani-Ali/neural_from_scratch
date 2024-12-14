import numpy as np


def tanh(net: np.ndarray) -> np.ndarray:
    """
    Applies the Hyperbolic Tangent (Tanh) activation function.

    Parameters:
    net (np.ndarray): Input array, can be of any shape.

    Returns:
    np.ndarray: Output array after applying the Tanh function.
    """
    # np.tanh computes the hyperbolic tangent element-wise
    return np.tanh(net)


def tanh_derivative(net: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the Hyperbolic Tangent (Tanh) activation function.

    Parameters:
    net (np.ndarray): Input array, can be of any shape.

    Returns:
    np.ndarray: Output array of derivatives of the Tanh function.
    """
    # Derivative of tanh: 1 - (tanh(net) ** 2)
    return 1 - (np.tanh(net) ** 2)
