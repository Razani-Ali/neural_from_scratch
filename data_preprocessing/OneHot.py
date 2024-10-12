import numpy as np


def one_hot_encoder(Y_data: np.ndarray):
    """
    Converts categorical labels into one-hot encoded format.

    Parameters:
    Y_data (np.ndarray): 1D or 2D array of categorical labels.

    Returns:
    tuple[np.ndarray, np.ndarray]: 
        - One-hot encoded array.
        - Array of unique labels.
    """
    # Find unique labels and map original labels to these unique labels
    unique_labels, labels_mapping = np.unique(Y_data.ravel(), return_inverse=True)
    
    # Create an identity matrix of size equal to the number of unique labels
    identity_matrix = np.eye(len(unique_labels))
    
    # Use labels mapping to index into the identity matrix to get one-hot encoded result
    one_hot_encoded = identity_matrix[labels_mapping]
    
    return one_hot_encoded, unique_labels