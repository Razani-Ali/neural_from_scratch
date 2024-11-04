import numpy as np
from initializers.weight_initializer import RBF_weight_init
from optimizers.set_optimizer import init_optimizer


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

    def all_params(self) -> int:
        """
        Calculate the number of parameters (centers and variances).

        Returns:
        --------
        int:
            The total number of parameters.
        """
        return np.size(self.center) + np.size(self.var)

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
    
    def optimzer_init(self, optimizer='Adam', **kwargs) -> None:
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, grads: np.ndarray, learning_rate: float = 1e-3) -> None:
        deltas = self.Optimizer(grads, learning_rate)
        ind2 = 0
        if self.train_center:
            ind1 = ind2
            ind2 += int(np.size(self.center))
            delta_center = deltas[ind1:ind2].reshape(self.center.shape)
            self.center -= delta_center
        if self.train_var:
            ind1 = ind2
            ind2 += np.size(self.var)
            delta_var = deltas[ind1:ind2].reshape(self.var.shape)
            self.var -= delta_var

    #################################################################

    def backward(self, error_batch: np.ndarray, learning_rate: float = 1e-3, return_error=False, return_grads=False, modify=True):
        # Initialize arrays for output error and gradients
        if return_error:
            error_in = np.zeros(self.input.shape)
        
        # Initialize the gradients for weights and biases
        grad_cen = np.zeros(self.center.shape) if self.train_center else None
        grad_var = np.zeros(self.var.shape) if self.train_var else None

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
            if return_error:
                error_x = (one_batch_error.ravel() * self.output[batch_index].ravel() * self.var.ravel() ** -2).reshape((-1, 1))
                error_in[batch_index] = (np.ones((self.input_size, self.output_size)) @ error_x).ravel() *\
                    (np.sum(2 * self.center, axis=0).ravel() - 4 * self.input[batch_index].ravel())

        # Normalize gradients by batch size
        if self.train_center:
            grad_cen /= error_batch.shape[0]
        if self.train_var:
            grad_var /= error_batch.shape[0]

        # Update the parameters using the computed gradients
        grads = None if (grad_cen is None) and (grad_var is None) else np.array([]).reshape((-1,1))
        if grads is not None:
            if grad_cen is not None:
                grads = np.concatenate((grads, grad_cen.reshape((-1,1))))
            if grad_var is not None:
                grads = np.concatenate((grads, grad_var.reshape((-1,1))))
        if modify:
            self.update(grads, learning_rate=learning_rate)

        # Return parameters gradients and gradient with respect to input if needed
        if return_error and return_grads:
            return {'error_in': error_in, 'gradients': grads}
        elif return_error:
            return error_in
        elif return_grads:
            return grads
