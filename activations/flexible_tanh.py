import numpy as np



def flex_tanh(a: np.ndarray, net: np.ndarray) -> np.ndarray:
    """
    Flexible tanh activation function that applies different scaling factors for each input.
    
    Parameters:
    a (np.ndarray): Scaling factors (vector) for each element in the 'net' array.
    net (np.ndarray): Input array (vector) for which the flexible tanh activation is computed.
    
    Returns:
    np.ndarray: Output array where the flexible tanh activation has been applied element-wise.
    """
    out = np.zeros(net.shape)  # Initialize the output array
    for i in range(len(a)):
        if np.abs(a[i]) < 0.001:  # For very small 'a', use linear approximation to avoid division by near-zero values
            out[i] = 0.5 * net[i]  # Linear approximation: 0.5 * net
        else:
            # Calculate the flexible tanh activation for larger values of 'a'
            out[i] = 1 / a[i] * (1 - np.exp(-net[i] * a[i])) / (1 + np.exp(-net[i] * a[i]))
    return out


def flex_tanh_star_derivative(a: np.ndarray, net: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the flexible tanh function with respect to the scaling factor 'a'.
    
    Parameters:
    a (np.ndarray): Scaling factors (vector) for each element in the 'net' array.
    net (np.ndarray): Input array (vector) for which the flexible tanh activation was computed.
    
    Returns:
    np.ndarray: Derivative of the flexible tanh function w.r.t 'a' for each element.
    """
    out = np.zeros(net.shape)  # Initialize the output array
    for i in range(len(a)):
        if np.abs(a[i]) < 0.001:
            out[i] = 0  # Derivative w.r.t 'a' is 0 when 'a' is very small
        else:
            # Calculate the hyperbolic tangent value (flexible tanh)
            g = (1 - np.exp(-net[i] * a[i])) / (1 + np.exp(-net[i] * a[i]))
            
            # Compute the derivative of tanh w.r.t 'a'
            dg_da = (2 * net[i] * np.exp(-net[i] * a[i])) / (1 + np.exp(-net[i] * a[i]))**2
            
            # Derivative calculation
            out[i] = -1 / a[i]**2 * g + 1 / a[i] * dg_da
    return out


def flex_tanh_derivative(a: np.ndarray, net: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the flexible tanh function with respect to 'net' (input).
    
    Parameters:
    a (np.ndarray): Scaling factors (vector) for each element in the 'net' array.
    net (np.ndarray): Input array (vector) for which the flexible tanh activation was computed.
    
    Returns:
    np.ndarray: Derivative of the flexible tanh function w.r.t 'net' for each element.
    """
    out = np.zeros(net.shape)  # Initialize the output array
    for i in range(len(a)):
        if np.abs(a[i]) < 0.001:
            out[i] = 0.5  # Derivative w.r.t 'net' is 0.5 when 'a' is small (linear approximation)
        else:
            # Derivative of tanh w.r.t net
            out[i] = 2 * np.exp(-net[i] * a[i]) / (1 + np.exp(-net[i] * a[i]))**2
    return out
