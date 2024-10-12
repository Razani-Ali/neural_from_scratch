import numpy as np
from sklearn.cluster import KMeans
from initializers.orthogonal import orthogonal_initialization
from initializers.xavier import xavier_init
from initializers.he import he_init
from initializers.uniform import uniform_init
from initializers.lecun import lecun_init

############################################################################################################################# Dense

def Dense_weight_init(input_size: int, output_size: int, method: str = 'xavier', 
                      distribution: str = 'normal', scale_factor: float = 1.0, 
                      ranges: tuple = None) -> np.ndarray:
    """
    Initializes the weights for a dense (fully connected) layer.

    Parameters:
    input_size (int): Number of input neurons.
    output_size (int): Number of output neurons.
    method (str): Initialization method ('xavier', 'he', 'uniform', 'lecun', 'orthogonal').
    distribution (str): Distribution type ('normal', 'uniform').
    scale_factor (float): Scaling factor for certain initialization methods.
    ranges (tuple, optional): Range (min, max) for uniform initialization method.

    Returns:
    np.ndarray: Initialized weight matrix of shape (output_size, input_size).
    """
    # Define the shape of the weight matrix
    shape = (output_size, input_size)
    
    if method == 'xavier':
        # Xavier initialization
        W = xavier_init(shape, distribution=distribution)
    
    elif method == 'he':
        # He initialization
        W = he_init(shape, distribution=distribution)
    
    elif method == 'uniform':
        # Uniform initialization with custom or default range
        W = uniform_init(shape, ranges=ranges)
    
    elif method == 'lecun':
        # LeCun initialization
        W = lecun_init(shape, distribution=distribution)
    
    elif method == 'orthogonal':
        # Orthogonal initialization
        W = orthogonal_initialization(shape, gain=scale_factor)
    
    else:
        # Unsupported initialization method
        raise ValueError('Your initialization method is not supported')
    
    return W

############################################################################################################################# RBF


def RBF_weight_init(input_size: int, output_size: int, method: str = 'random', distribution: str = 'uniform', 
                    ranges: tuple = (0, 1), var: float = 1.0, data: np.ndarray = None) -> np.ndarray:
    """
    Initialize the weights (centers) for an RBF layer.

    Parameters:
    -----------
    input_size : int
        The number of input features (dimensionality of each data point).
        
    output_size : int
        The number of RBF neurons (output size).
        
    method : str, optional
        Method for weight initialization. Choices are:
        - 'random': Random initialization (default).
        - 'zeros': Initialize centers with zeros.
        - 'Kmeans': Use K-means clustering to initialize centers.
        
    distribution : str, optional
        Distribution to use when method is 'random'. Choices are:
        - 'uniform': Uniform distribution (default).
        - 'normal': Gaussian distribution.
        
    ranges : tuple, optional
        Range for the uniform distribution when `distribution` is 'uniform'. Default is (0, 1).
        
    var : float, optional
        Variance to use when `distribution` is 'normal'. Default is 1.0.
        
    data : np.ndarray, optional
        Training data for K-means initialization. Required if `method` is 'Kmeans'.
        Shape should be (n_samples, input_size).
        
    Returns:
    --------
    W : np.ndarray
        The initialized weight matrix (centers) with shape (output_size, input_size).
        Each row represents the center of an RBF neuron.
    
    Raises:
    -------
    ValueError
        If unsupported initialization methods or distributions are specified, or if data is not provided for K-means.
    """
    
    # Shape of the weight matrix (output_size, input_size), where each row is a center for an RBF neuron
    shape = (output_size, input_size)

    # Random initialization methods
    if method == 'random':
        # Uniform distribution in the given range
        if distribution == 'uniform':
            W = np.random.uniform(ranges[0], ranges[1], shape)
        # Gaussian distribution with zero mean and given variance
        elif distribution == 'normal':
            W = np.random.normal(0.0, np.sqrt(var), shape)
        else:
            raise ValueError('Your distribution is not supported. Choose "uniform" or "normal".')
    
    # Zero initialization
    elif method == 'zeros':
        W = np.zeros(shape)
    
    # K-means clustering initialization
    elif method == 'Kmeans':
        if data is None:
            raise ValueError("Data must be provided for K-means initialization.")
        
        # Perform K-means clustering to find cluster centers
        kmeans = KMeans(n_clusters=output_size)
        kmeans.fit(data)
        W = kmeans.cluster_centers_
    
    # If an unsupported method is provided
    else:
        raise ValueError('Your initialization method is not supported. Choose "random", "zeros", or "Kmeans".')
    
    return W
