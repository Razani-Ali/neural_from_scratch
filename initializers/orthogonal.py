import numpy as np


def orthogonal_initialization(shape: tuple, gain: float = 1.0) -> np.ndarray:
    """
    Perform orthogonal initialization of weights.

    Parameters:
    shape (tuple): A tuple specifying the shape (num_rows, num_cols) of the weight matrix to initialize.
    gain (float): A scaling factor for the orthogonal matrix. Default is 1.0.

    Returns:
    np.ndarray: An orthogonally initialized weight matrix of the given shape.
    """
    num_rows, num_cols = shape
    
    # Create a random matrix of shape (num_rows, num_cols) from a standard normal distribution
    a = np.random.randn(num_rows, num_cols)
    
    # Perform Singular Value Decomposition (SVD) on the random matrix
    u, _, v = np.linalg.svd(a, full_matrices=False)
    
    # Choose the orthogonal matrix that has the same number of rows as the original matrix
    q = u if num_rows > num_cols else v
    
    # Scale the orthogonal matrix by the gain factor
    q = q * gain
    
    # Return the orthogonal matrix with the desired shape
    return q[:num_rows, :num_cols]