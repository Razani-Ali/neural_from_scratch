import numpy as np
from initializers.weight_initializer import RBF_weight_init


class RBF:
    """
    Radial Basis Function (RBF) Neural Network layer.

    Parameters:
    -----------
    input_size : int
        The dimension of the input data.
    output_size : int
        The number of neurons in the RBF layer (also the dimension of the output).
    batch_size : int, optional
        The number of samples to be processed in a batch (default is 32).
    center_init_method : str, optional
        The method for initializing the centers of the RBF neurons. Options are 'random', 'zeros', and 'Kmeans' (default is 'random').
    train_center : bool, optional
        If True, the centers of the RBF neurons are trainable (default is True).
    train_var : bool, optional
        If True, the variance of the RBF neurons is trainable (default is True).
    center_distribution : str, optional
        The distribution used for initializing the centers ('normal' or 'uniform') (default is 'normal').
    data : np.ndarray, optional
        If 'Kmeans' initialization is used, this is the dataset to fit the K-means (default is None).
    var_init_method : str, optional
        The method to initialize the variances of the RBF neurons. Options are 'constant', 'average', and 'max' (default is 'average').
    var_init_const : float, optional
        The constant value used to initialize variances if var_init_method is 'constant' (default is 1).
    center_uniform_range : tuple, optional
        Range of values for uniform distribution initialization of centers (default is None).
    center_normal_var : float, optional
        The variance for normal distribution initialization of centers (default is 1).

    Attributes:
    -----------
    center : np.ndarray
        The centers of the RBF neurons.
    var : np.ndarray
        The variances (spread) of the RBF neurons.
    net : np.ndarray
        The calculated net input (distance between input and centers).
    output : np.ndarray
        The final output (activations) of the RBF neurons for the input batch.

    """
    def __init__(self, input_size: int, output_size: int, batch_size: int = 32,
                 center_init_method: str = 'random', train_center: bool = True, train_var: bool = True,
                 center_distribution: str = 'normal', data=None, var_init_method='average', var_init_const=1,
                 center_uniform_range: tuple = None, center_normal_var: float = 1):

        # Initialize input/output sizes and training flags
        self.output_size = output_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.train_center = train_center
        self.train_var = train_var

        # Initialize centers using RBF_weight_init method
        self.center = RBF_weight_init(input_size, output_size, method=center_init_method,
                                      distribution=center_distribution, ranges=center_uniform_range,
                                      var=center_normal_var, data=data)

        # Initialize variances based on the chosen method
        if var_init_method == 'constant':
            self.var = np.zeros((output_size, 1)) + var_init_const
        elif var_init_method == 'average':
            self.var = np.mean(self.center, axis=1).reshape((-1, 1)) / output_size
        elif var_init_method == 'max':
            self.var = np.max(self.center, axis=1).reshape((-1, 1)) / np.sqrt(2 * output_size)

        # Initialize intermediate arrays for storing results
        self.net = np.zeros((batch_size, output_size, 1))
        self.output = np.zeros((batch_size, output_size, 1))

    #################################################################

    def trainable_params(self) -> int:
        """
        Calculate the number of trainable parameters (centers and variances).

        Returns:
        --------
        int:
            The total number of trainable parameters.
        """
        params = 0
        if self.train_center:
            params += np.size(self.center)
        if self.train_var:
            params += np.size(self.var)
        return params

    #################################################################

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Forward pass of the RBF layer.

        Parameters:
        -----------
        input : np.ndarray
            Input batch of shape (batch_size, input_size).

        Returns:
        --------
        np.ndarray:
            Output of the RBF layer (batch_size, output_size).
        """

        # Reshape input for consistency
        input = input.reshape((-1, self.input_size))

        # Store input for backward pass
        self.input = input

        # Check if batch size is valid
        if self.batch_size < input.shape[0]:
            raise ValueError('Data batch size cannot be larger than model batch size')

        # Calculate distance between input and centers
        for batch_index, input_vector in enumerate(input):
            self.net[batch_index] = np.linalg.norm((np.repeat(input_vector.reshape(1, -1),
                                                              self.output_size, axis=0) - self.center), axis=1).reshape((-1, 1))
            try:
                # Calculate the RBF output using Gaussian kernel
                self.output[batch_index] = np.exp(-0.5 * np.square(self.net[batch_index].ravel() / self.var.ravel())).reshape(-1, 1)
            except:
                # Avoid division by zero
                self.output[batch_index] = np.exp(-0.5 * np.square(self.net[batch_index].ravel() / (self.var.ravel()+1e-7))).reshape(-1, 1)

        return self.output[:input.shape[0]].reshape((-1, self.output_size))

    #################################################################

    def Adam_init(self):
        """
        Initialize the Adam optimizer's moment vectors.
        """
        self.t = 0  # Time step for Adam optimizer

        if self.train_center:
            self.center_mt = np.zeros(self.center.shape)  # First moment (mean) for centers
            self.center_vt = np.zeros(self.center.shape)  # Second moment (variance) for centers

        if self.train_var:
            self.var_mt = np.zeros(self.var.shape)  # First moment (mean) for variances
            self.var_vt = np.zeros(self.var.shape)  # Second moment (variance) for variances

    #################################################################

    def update(self, grad_cen: np.ndarray, grad_var: np.ndarray, method: str = 'Adam', learning_rate: float = 1e-3,
               var_learning_rate: float = 2e-4, adam_beta1: float = 0.9, adam_beta2: float = 0.99):
        """
        Update the RBF parameters (centers and variances) using the selected optimizer.

        Parameters:
        -----------
        grad_cen : np.ndarray
            Gradient of the loss with respect to the centers.
        grad_var : np.ndarray
            Gradient of the loss with respect to the variances.
        method : str, optional
            Optimization method ('Adam' or 'SGD') (default is 'Adam').
        learning_rate : float, optional
            Learning rate for center updates (default is 1e-3).
        var_learning_rate : float, optional
            Learning rate for variance updates (default is 2e-4).
        adam_beta1 : float, optional
            Beta1 parameter for Adam optimizer (default is 0.9).
        adam_beta2 : float, optional
            Beta2 parameter for Adam optimizer (default is 0.99).
        """

        if method == 'Adam':
            eps = 1e-7
            self.t += 1  # Increment time step

            if self.train_center:
                # Adam update for centers
                self.center_mt = adam_beta1 * self.center_mt + (1 - adam_beta1) * grad_cen
                self.center_vt = adam_beta2 * self.center_vt + (1 - adam_beta2) * np.square(grad_cen)
                m_hat_cen = self.center_mt / (1 - adam_beta1 ** self.t)
                v_hat_cen = self.center_vt / (1 - adam_beta2 ** self.t)
                delta_cen = learning_rate * m_hat_cen / (np.sqrt(v_hat_cen) + eps)

            if self.train_var:
                # Adam update for variances
                self.var_mt = adam_beta1 * self.var_mt + (1 - adam_beta1) * grad_var
                self.var_vt = adam_beta2 * self.var_vt + (1 - adam_beta2) * np.square(grad_var)
                m_hat_var = self.var_mt / (1 - adam_beta1 ** self.t)
                v_hat_var = self.var_vt / (1 - adam_beta2 ** self.t)
                delta_var = var_learning_rate * m_hat_var / (np.sqrt(v_hat_var) + eps)
        else:
            # SGD update
            delta_cen = learning_rate * grad_cen
            if self.train_var:
                delta_var = var_learning_rate * grad_var

        # Update parameters
        if self.train_center:
            self.center -= delta_cen
        if self.train_var:
            self.var -= delta_var

    #################################################################

    def backward(self, error_batch: np.ndarray, method: str = 'Adam', 
                 learning_rate: float = 1e-3, var_learning_rate: float = 2e-4, 
                 adam_beta1: float = 0.9, adam_beta2: float = 0.99) -> np.ndarray:
        """
        Backpropagate the error through the RBF layer.

        Parameters:
        -----------
        error_batch : np.ndarray
            The error signal for the current batch.
        method : str, optional
            The optimization method ('Adam' or 'SGD') (default is 'Adam').
        learning_rate : float, optional
            Learning rate for centers (default is 1e-3).
        var_learning_rate : float, optional
            Learning rate for variances (default is 2e-4).
        adam_beta1 : float, optional
            Beta1 parameter for Adam optimizer (default is 0.9).
        adam_beta2 : float, optional
            Beta2 parameter for Adam optimizer (default is 0.99).

        Returns:
        --------
        np.ndarray:
            Gradient of the error with respect to the input.
        """

        # Initialize arrays for output error and gradients
        error_out = np.zeros(self.input.shape)
        grad_cen = None
        if self.train_center:
            grad_cen = np.zeros(self.center.shape)
        grad_var = None
        if self.train_var:
            grad_var = np.zeros(self.var.shape)

        # Iterate over each batch and calculate gradients
        for batch_index, one_batch_error in enumerate(error_batch):

            if self.train_center:
                # Gradient w.r.t. centers
                grad_cen = np.diag(one_batch_error.ravel() * self.output[batch_index].ravel() * self.var.ravel() ** -2) @\
                            (np.repeat(self.input[batch_index].reshape(1, -1), self.output_size, axis=0) - self.center)
                
            if self.train_var:
                # Gradient w.r.t. variances
                grad_var += (one_batch_error.ravel() * self.net[batch_index].ravel() ** 2 *\
                    self.var.ravel() ** -3 * self.output[batch_index].ravel()).reshape((-1, 1))
            
            # Error backpropagated to the input
            error_x = (one_batch_error.ravel() * self.output[batch_index].ravel() * self.var.ravel() ** -2).reshape((-1, 1))
            error_out[batch_index] = (np.ones((self.input_size, self.output_size)) @ error_x).ravel() *\
                (np.sum(2 * self.center, axis=0).ravel() - 4 * self.input[batch_index].ravel())

        # Normalize gradients by batch size
        if self.train_center:
            grad_cen /= error_batch.shape[0]
        if self.train_var:
            grad_var /= error_batch.shape[0]

        # Update parameters using the chosen optimization method
        self.update(grad_cen, grad_var, method=method, learning_rate=learning_rate,
                var_learning_rate=var_learning_rate, adam_beta1=adam_beta1, adam_beta2=adam_beta2)

        return error_out
