import numpy as np


def lecun_init(shape: tuple, distribution: str = 'normal') -> np.ndarray:
    """
    LeCun initialization for dense layers.

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
        # LeCun normal initialization
        mean = 0
        var = 1 / input_size  # Variance based on LeCun initialization formula
        W = np.random.normal(mean, np.sqrt(var), shape)  # Generate normally distributed weights
    elif distribution == 'uniform':
        # LeCun uniform initialization
        maximum = np.sqrt(3 / input_size)  # Calculate max range
        W = np.random.uniform(-maximum, maximum, shape)  # Generate uniformly distributed weights
    else:
        raise ValueError('Unsupported distribution for LeCun initialization')  # Handle unsupported distribution

    return W
