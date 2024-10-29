import numpy as np


def selu(net: np.ndarray, lambda_: float, alpha: float) -> np.ndarray:
    """
    Scaled Exponential Linear Unit (SELU) activation function.

    Parameters:
    net (np.ndarray): Input array (vector) for which the SELU activation is computed.
    lambda_ (float): Scaling factor that normalizes the output.
    alpha (float): Scaling factor for negative inputs, controlling the exponential curve.

    Returns:
    np.ndarray: Output array where the SELU activation has been applied element-wise.
    """
    # Apply SELU: positive inputs scaled by 'lambda_', negative inputs follow the exponential curve with 'alpha'
    return np.where(net > 0, lambda_ * net, lambda_ * alpha * (np.exp(net) - 1))


def selu_star_derivative(net: np.ndarray, lambda_: float, alpha: float) -> np.ndarray:
    """
    Computes the derivative of the SELU function with respect to the scaling factor 'alpha'.

    Parameters:
    net (np.ndarray): Input array (vector) for which the SELU activation was computed.
    lambda_ (float): Scaling factor that normalizes the output.
    alpha (float): Scaling factor for negative inputs, controlling the exponential curve.

    Returns:
    np.ndarray: Derivative of the SELU function w.r.t. 'alpha' for each element.
    """
    # Derivative w.r.t. 'alpha': only affects negative inputs (net <= 0)
    return np.where(net > 0, 0, lambda_ * (np.exp(net) - 1))


def selu_derivative(net: np.ndarray, lambda_: float, alpha: float) -> np.ndarray:
    """
    Computes the derivative of the SELU function with respect to 'net' (input).

    Parameters:
    net (np.ndarray): Input array (vector) for which the SELU activation was computed.
    lambda_ (float): Scaling factor that normalizes the output.
    alpha (float): Scaling factor for negative inputs, controlling the exponential curve.

    Returns:
    np.ndarray: Derivative of the SELU function w.r.t. 'net' for each element.
    """
    # Derivative w.r.t. 'net': linear for positive inputs, exponential for negative inputs
    return np.where(net > 0, lambda_, lambda_ * alpha * np.exp(net))


def selu_lambda_derivative(net: np.ndarray, lambda_: float, alpha: float) -> np.ndarray:
    """
    Computes the derivative of the SELU function with respect to the scaling factor 'lambda_'.

    Parameters:
    net (np.ndarray): Input array (vector) for which the SELU activation was computed.
    lambda_ (float): Scaling factor that normalizes the output.
    alpha (float): Scaling factor for negative inputs, controlling the exponential curve.

    Returns:
    np.ndarray: Derivative of the SELU function w.r.t. 'lambda_' for each element.
    """
    # Derivative w.r.t. 'lambda_': directly proportional to the input for positive values,
    # and the exponential part for negative values
    return np.where(net > 0, net, alpha * (np.exp(net) - 1))
