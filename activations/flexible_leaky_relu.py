import numpy as np

def flex_leaky_relu(net: np.ndarray, alpha: float) -> np.ndarray:
    """
    Leaky ReLU activation function with a fixed negative slope factor.

    Parameters:
    net (np.ndarray): Input array (vector) for which the Leaky ReLU activation is computed.
    alpha (float): Negative slope factor for the Leaky ReLU, determines how much leakage is allowed for negative values.

    Returns:
    np.ndarray: Output array where the Leaky ReLU activation has been applied element-wise.
    """
    # Apply Leaky ReLU: positive inputs remain unchanged, negative inputs are scaled by 'alpha'
    return np.where(net > 0, net, alpha * net)


def flex_leaky_relu_star_derivative(net: np.ndarray, alpha: float) -> np.ndarray:
    """
    Computes the derivative of the Leaky ReLU function with respect to 'alpha'.

    Parameters:
    net (np.ndarray): Input array (vector) for which the Leaky ReLU activation was computed.
    alpha (float): Negative slope factor for the Leaky ReLU.

    Returns:
    np.ndarray: Derivative of the Leaky ReLU function w.r.t. 'alpha' for each element.
    """
    # Derivative w.r.t. 'alpha': 0 for positive inputs, 'net' for negative inputs
    return np.where(net > 0, 0, net)


def flex_leaky_relu_derivative(net: np.ndarray, alpha: float) -> np.ndarray:
    """
    Computes the derivative of the Leaky ReLU function with respect to 'net' (input).

    Parameters:
    net (np.ndarray): Input array (vector) for which the Leaky ReLU activation was computed.
    alpha (float): Negative slope factor for the Leaky ReLU.

    Returns:
    np.ndarray: Derivative of the Leaky ReLU function w.r.t. 'net' for each element.
    """
    # Derivative w.r.t. 'net': 1 for positive inputs, 'alpha' for negative inputs
    return np.where(net > 0, 1, alpha)
