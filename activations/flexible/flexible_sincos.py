import numpy as np


def sin_plus_cos_flexible(net: np.ndarray, alpha: float, lambda_: float) -> np.ndarray:
    """
    Flexible Sine+Cosine activation function: sin(alpha * net + lambda) + cos(alpha * net + lambda).

    Parameters:
    net (np.ndarray): Input array for which the Sine+Cosine activation is computed.
    alpha (float): Scaling factor for the input.
    lambda_ (float): Translation factor added to the input.

    Returns:
    np.ndarray: Output after applying the Sine+Cosine activation element-wise.
    """
    return np.sin(alpha * net + lambda_) + np.cos(alpha * net + lambda_)

def sin_plus_cos_flexible_derivative(net: np.ndarray, alpha: float, lambda_: float) -> np.ndarray:
    """
    Derivative of Flexible Sine+Cosine activation function with respect to 'net'.

    Parameters:
    net (np.ndarray): Input array for which the derivative is computed.
    alpha (float): Scaling factor for the input.
    lambda_ (float): Translation factor added to the input.

    Returns:
    np.ndarray: Derivative with respect to 'net'.
    """
    return alpha * (np.cos(alpha * net + lambda_) - np.sin(alpha * net + lambda_))

def sin_plus_cos_flexible_alpha_derivative(net: np.ndarray, alpha: float, lambda_: float) -> np.ndarray:
    """
    Derivative of Flexible Sine+Cosine activation function with respect to 'alpha'.

    Parameters:
    net (np.ndarray): Input array for which the derivative is computed.
    alpha (float): Scaling factor for the input.
    lambda_ (float): Translation factor added to the input.

    Returns:
    np.ndarray: Derivative with respect to 'alpha'.
    """
    return net * (np.cos(alpha * net + lambda_) - np.sin(alpha * net + lambda_))

def sin_plus_cos_flexible_lambda_derivative(net: np.ndarray, alpha: float, lambda_: float) -> np.ndarray:
    """
    Derivative of Flexible Sine+Cosine activation function with respect to 'lambda_'.

    Parameters:
    net (np.ndarray): Input array for which the derivative is computed.
    alpha (float): Scaling factor for the input.
    lambda_ (float): Translation factor added to the input.

    Returns:
    np.ndarray: Derivative with respect to 'lambda_'.
    """
    return np.cos(alpha * net + lambda_) - np.sin(alpha * net + lambda_)
