import numpy as np


def softmax(net: np.ndarray) -> np.ndarray:
    """
    Applies the Softmax activation function.

    Parameters:
    net (np.ndarray): Input array, can be of any shape.

    Returns:
    np.ndarray: Output array after applying the Softmax function. The values in the output array sum to 1.
    """
    # Subtracting the maximum value of net to prevent numerical overflow
    net -= np.max(net)
    # Compute the exponent of each element
    exp_z = np.exp(net)
    # Compute the softmax probabilities
    softmax_probs = exp_z / np.sum(exp_z)
    return softmax_probs
    

def softmax_derivative(net: np.ndarray) -> np.ndarray:
    """
    Computes the derivative of the Softmax activation function.

    Parameters:
    net (np.ndarray): Input array, can be of any shape.

    Returns:
    np.ndarray: Output array of derivatives of the Softmax function.
    """
    # Compute softmax probabilities
    z = softmax(net)
    # Size of the input array (number of elements along the first axis)
    n = np.size(z, axis=0)
    # Create a matrix where each column is the softmax output (repeated)
    M = np.repeat(z, n, axis=1)
    # Identity matrix of size n
    I = np.eye(n)
    # Derivative of softmax: M * (I - M)
    return M * (I - M)