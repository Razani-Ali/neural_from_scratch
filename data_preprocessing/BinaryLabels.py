import numpy as np

def binary_label(Y_data: np.ndarray):
    """
    Converts categorical labels into a binary encoded format with values 1.0 and 0.0.

    Parameters:
    Y_data (np.ndarray): 1D or 2D array of categorical labels.

    Returns:
    tuple[np.ndarray, np.ndarray]: 
        - Binary encoded array with values 1.0 and 0.0.
        - Array of unique labels, where the first label corresponds to 0.0 and the second to 1.0.
    """
    # Find unique labels and verify that there are exactly two unique labels
    unique_labels = np.unique(Y_data.ravel())
    if len(unique_labels) != 2:
        raise ValueError("Input data must contain exactly two unique labels for binary encoding.")

    # Create a mapping to assign 0.0 to the first label and 1.0 to the second
    binary_encoded = np.where(Y_data.ravel() == unique_labels[0], 0.0, 1.0)

    return binary_encoded.reshape(Y_data.shape), unique_labels
