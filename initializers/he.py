import numpy as np


def he_init(shape: tuple, distribution: str = 'normal') -> np.ndarray:
    """
    He initialization for dense layers specially for Relu activation functions.

    Parameters:
    -----------
    shape (tuple): A tuple specifying the shape (num_rows, num_cols) of the weight matrix to initialize.
    distribution : str, optional
        Distribution type ('normal' or 'uniform'). Default is 'normal'.

    Returns:
    --------
    np.ndarray:
        Initialized weight matrix of shape (output_size, input_size).
    """
    _, input_size = shape

    if distribution == 'normal':
        # He normal initialization
        mean = 0
        var = 2 / input_size  # Variance based on He initialization formula
        W = np.random.normal(mean, np.sqrt(var), shape)  # Generate normally distributed weights
    elif distribution == 'uniform':
        # He uniform initialization
        maximum = np.sqrt(6 / input_size)  # Calculate max range
        W = np.random.uniform(-maximum, maximum, shape)  # Generate uniformly distributed weights
    else:
        raise ValueError('Unsupported distribution for He initialization')  # Handle unsupported distribution

    return W
