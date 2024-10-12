import numpy as np


def sliding_window(time_series_vector: np.ndarray, num_column: int) -> np.ndarray:
    """
    Generates a sliding window view of the input time series data.

    Parameters:
    time_series_vector (np.ndarray): 1D array representing the time series data.
    num_column (int): Number of columns (window size).

    Returns:
    np.ndarray: 2D array where each row is a window of the time series data.
    """
    # Ensure the input time series is a 1D array
    time_series_vector = np.ravel(time_series_vector)
    
    # Determine the number of rows for the output based on the window size
    num_rows = np.size(time_series_vector, axis=0)
    
    # Initialize the output array with zeros
    seriesed_data = np.zeros((num_rows - num_column + 1, num_column))
    
    # Populate the output array with the sliding window data
    for i in range(num_column):
        seriesed_data[:, i] = time_series_vector[i:num_rows - num_column + i + 1]
    
    return seriesed_data