import numpy as np


def uniform_init(shape: tuple, ranges: tuple = None) -> np.ndarray:
    """
    Uniform initialization for dense layers with custom or default range.

    Parameters:
    -----------
    shape (tuple): A tuple specifying the shape (num_rows, num_cols) of the weight matrix to initialize.
    ranges : tuple, optional
        Custom range (min, max) for the uniform distribution. Default is None, which 
        uses [-sqrt(1/input_size), sqrt(1/input_size)].

    Returns:
    --------
    np.ndarray:
        Initialized weight matrix of shape (output_size, input_size).
    """
    _, input_size = shape

    if ranges is None:
        # Use default range based on input size
        k = 1 / input_size  # Scaling factor
        maximum = np.sqrt(k)  # Calculate max range
        minimum = -maximum  # Calculate min range
    else:
        # Use custom range
        minimum, maximum = ranges

    # Generate uniformly distributed weights within the specified range
    W = np.random.uniform(minimum, maximum, shape)

    return W
