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

def RBF_init(
    input_size: int, 
    output_size: int, 
    method: str = 'random', 
    distribution: str = 'uniform', 
    ranges: tuple = (0, 1), 
    var: float = 1.0, 
    data: np.ndarray = None, 
    var_init_method: str = 'constant', 
    var_constant: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initializes the centers and variances for Radial Basis Function (RBF) neurons.

    Parameters:
    ----------
    input_size : int
        The number of input dimensions for each RBF neuron.
    output_size : int
        The number of RBF neurons.
    method : str, optional (default='random')
        The method to initialize the centers. Options:
        - 'random': Random initialization.
        - 'zeros': Initialize centers with zeros.
        - 'Kmeans': Use k-means clustering to initialize centers.
    distribution : str, optional (default='uniform')
        Distribution to use for random initialization. Options:
        - 'uniform': Uniform distribution within the specified range.
        - 'normal': Normal distribution with variance `var`.
    ranges : tuple, optional (default=(0, 1))
        The range for uniform distribution initialization.
    var : float, optional (default=1.0)
        Variance for normal distribution initialization.
    data : np.ndarray, optional (default=None)
        Data to be used for k-means initialization or variance calculation based on data clusters.
    var_init_method : str, optional (default='constant')
        Method to initialize variances. Options:
        - 'zeros': All variances set to zero.
        - 'constant': All variances set to `var_constant`.
        - 'random_uniform': Random values from a uniform distribution.
        - 'random_normal': Random values from a normal distribution.
        - 'mean': Row-wise mean of the centers.
        - 'max': Row-wise maximum of the centers.
        - 'Kmeans_cluster_var': Variance of data within each k-means cluster.
        - 'Kmeans_max_dis': Maximum distance of data points within each cluster to its center.
        - 'Kmeans_ave_dis': Average distance of data points within each cluster to its center.
        - 'min_c2c': Minimum Euclidean distance between centers.
        - 'mean_c2c': Mean Euclidean distance between centers.
    var_constant : float, optional (default=1.0)
        Constant value to use when `var_init_method='constant'`.

    Returns:
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - Centers: A numpy array of shape (output_size, input_size) representing the initialized centers.
        - Variances: A numpy array of shape (output_size, 1) representing the initialized variances.

    Raises:
    ------
    ValueError
        If an unsupported method, distribution, or variance initialization method is specified.
    """

    # Shape of the weight matrix (output_size, input_size), where each row is a center for an RBF neuron
    shape = (output_size, input_size)

    # Initialize Centers
    if method == 'random':
        if distribution == 'uniform':
            Centers = np.random.uniform(ranges[0], ranges[1], shape)
        elif distribution == 'normal':
            Centers = np.random.normal(0.0, np.sqrt(var), shape)
        else:
            raise ValueError('Your distribution is not supported. Choose "uniform" or "normal".')

    elif method == 'zeros':
        Centers = np.zeros(shape)

    elif method == 'Kmeans':
        if data is None:
            raise ValueError("Data must be provided for K-means initialization.")

        kmeans = KMeans(n_clusters=output_size)
        kmeans.fit(data)
        Centers = kmeans.cluster_centers_
    else:
        raise ValueError('Your initialization method is not supported. Choose "random", "zeros", or "Kmeans".')

    # Initialize Variances
    variances = np.zeros((output_size, 1))

    if var_init_method == 'zeros':
        variances = np.zeros((output_size, 1))

    elif var_init_method == 'constant':
        variances = np.full((output_size, 1), var_constant)

    elif var_init_method == 'random_uniform':
        variances = np.abs(np.random.uniform(size=(output_size, 1)))

    elif var_init_method == 'random_normal':
        variances = np.abs(np.random.normal(size=(output_size, 1)))

    elif var_init_method == 'mean':
        variances = np.mean(Centers, axis=1, keepdims=True)

    elif var_init_method == 'max':
        variances = np.max(Centers, axis=1, keepdims=True)

    elif 'Kmeans' in var_init_method:
        if method != 'Kmeans' or data is None:
            raise ValueError("cluster_var variance initialization requires Kmeans center initialization and data.")

        if var_init_method == 'Kmeans_cluster_var':
            for i in range(output_size):
                cluster_data = data[kmeans.labels_ == i]
                variances[i] = np.var(cluster_data)

        if var_init_method == 'Kmeans_max_dis':
            for i in range(output_size):
                cluster_data = data[kmeans.labels_ == i]
                variances[i] = np.max(np.sqrt(np.sum((cluster_data - Centers[i])**2, axis=1)))

        if var_init_method == 'Kmeans_ave_dis':
            for i in range(output_size):
                cluster_data = data[kmeans.labels_ == i]
                variances[i] = np.mean(np.sqrt(np.sum((cluster_data - Centers[i])**2, axis=1)))

    elif var_init_method == 'min_c2c':
        # Minimum Euclidean distance between each center and all other centers
        distances = np.linalg.norm(Centers[:, np.newaxis] - Centers[np.newaxis, :], axis=2)
        np.fill_diagonal(distances, np.NaN)  # Exclude distance to self
        variances = np.nanmin(distances, axis=1, keepdims=True)

    elif var_init_method == 'mean_c2c':
        # Mean Euclidean distance between each center and all other centers
        distances = np.linalg.norm(Centers[:, np.newaxis] - Centers[np.newaxis, :], axis=2)
        np.fill_diagonal(distances, np.NaN)  # Exclude distance to self
        variances = np.nanmean(distances, axis=1, keepdims=True)

    else:
        raise ValueError('Your variance initialization method is not supported. Choose a valid method.')

    return Centers, variances
