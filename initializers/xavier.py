import numpy as np


def xavier_init(shape: tuple, distribution: str = 'normal') -> np.ndarray:
    """
    Xavier initialization for dense layers specially for tanh or sigmoid activation functions.

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
    output_size, input_size = shape

    if distribution == 'normal':
        # Xavier normal initialization
        mean = 0
        var = 2 / (input_size + output_size)  # Variance based on Xavier formula
        W = np.random.normal(mean, np.sqrt(var), shape)  # Generate normally distributed weights
    elif distribution == 'uniform':
        # Xavier uniform initialization
        maximum = np.sqrt(6 / (input_size + output_size))  # Calculate max range
        W = np.random.uniform(-maximum, maximum, shape)  # Generate uniformly distributed weights
    else:
        raise ValueError('Unsupported distribution for Xavier initialization')  # Handle unsupported distribution

    return W
