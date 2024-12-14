import numpy as np


def sine(net: np.ndarray) -> np.ndarray:
    """
    Applies the Sine activation function.

    Parameters:
    net (np.ndarray): Input array, can be of any shape.

    Returns:
    np.ndarray: Output array after applying the Sine function.
    """
    # np.sin computes the sine of each element in the array
    return np.sin(net)


def sine_derivative(net: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the Sine activation function.

    Parameters:
    net (np.ndarray): Input array, can be of any shape.

    Returns:
    np.ndarray: Output array of derivatives of the Sine function.
    """
    # Derivative of sine: cos(net)
    return np.cos(net)


def cosine(net: np.ndarray) -> np.ndarray:
    """
    Applies the Cosine activation function.

    Parameters:
    net (np.ndarray): Input array, can be of any shape.

    Returns:
    np.ndarray: Output array after applying the Cosine function.
    """
    # np.cos computes the cosine of each element in the array
    return np.cos(net)


def cosine_derivative(net: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the Cosine activation function.

    Parameters:
    net (np.ndarray): Input array, can be of any shape.

    Returns:
    np.ndarray: Output array of derivatives of the Cosine function.
    """
    # Derivative of cosine: -sin(net)
    return -np.sin(net)


def sine_plus_cosine(net: np.ndarray) -> np.ndarray:
    """
    Applies the Sine+Cosine activation function.

    Parameters:
    net (np.ndarray): Input array, can be of any shape.

    Returns:
    np.ndarray: Output array after applying the Sine+Cosine function.
    """
    # Sine+Cosine function: sin(net) + cos(net)
    return np.sin(net) + np.cos(net)


def sine_plus_cosine_derivative(net: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the Sine+Cosine activation function.

    Parameters:
    net (np.ndarray): Input array, can be of any shape.

    Returns:
    np.ndarray: Output array of derivatives of the Sine+Cosine function.
    """
    # Derivative of Sine+Cosine: cos(net) - sin(net)
    return np.cos(net) - np.sin(net)
