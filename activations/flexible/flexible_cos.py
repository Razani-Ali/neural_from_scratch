import numpy as np


def cos_flexible(net: np.ndarray, alpha: float, lambda_: float) -> np.ndarray:
    """
    Flexible Cosine activation function: cos(alpha * net + lambda).

    Parameters:
    net (np.ndarray): Input array for which the Cosine activation is computed.
    alpha (float): Scaling factor for the input.
    lambda_ (float): Translation factor added to the input.

    Returns:
    np.ndarray: Output after applying the Cosine activation element-wise.
    """
    return np.cos(alpha * net + lambda_)

def cos_flexible_derivative(net: np.ndarray, alpha: float, lambda_: float) -> np.ndarray:
    """
    Derivative of Flexible Cosine activation function with respect to 'net'.

    Parameters:
    net (np.ndarray): Input array for which the derivative is computed.
    alpha (float): Scaling factor for the input.
    lambda_ (float): Translation factor added to the input.

    Returns:
    np.ndarray: Derivative with respect to 'net'.
    """
    return -alpha * np.sin(alpha * net + lambda_)

def cos_flexible_alpha_derivative(net: np.ndarray, alpha: float, lambda_: float) -> np.ndarray:
    """
    Derivative of Flexible Cosine activation function with respect to 'alpha'.

    Parameters:
    net (np.ndarray): Input array for which the derivative is computed.
    alpha (float): Scaling factor for the input.
    lambda_ (float): Translation factor added to the input.

    Returns:
    np.ndarray: Derivative with respect to 'alpha'.
    """
    return -net * np.sin(alpha * net + lambda_)

def cos_flexible_lambda_derivative(net: np.ndarray, alpha: float, lambda_: float) -> np.ndarray:
    """
    Derivative of Flexible Cosine activation function with respect to 'lambda_'.

    Parameters:
    net (np.ndarray): Input array for which the derivative is computed.
    alpha (float): Scaling factor for the input.
    lambda_ (float): Translation factor added to the input.

    Returns:
    np.ndarray: Derivative with respect to 'lambda_'.
    """
    return -np.sin(alpha * net + lambda_)
