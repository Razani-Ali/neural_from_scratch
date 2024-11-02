import numpy as np
from initializers.weight_initializer import RBF_weight_init

class Rough_RBF:
    """
    A class representing a rough Radial Basis Function (RBF) network 
    with dual-layer structure (upper and lower networks) and 
    an optional adaptive blend factor (alpha).

    Attributes:
    -----------
    input_size : int
        The number of input features to the RBF network.
    output_size : int
        The number of output neurons (RBF units).
    batch_size : int, optional
        The number of samples in each batch (default is 32).
    train_center : bool, optional
        Whether to train the centers of the RBFs (default is True).
    train_var : bool, optional
        Whether to train the variances of the RBFs (default is True).
    train_alpha : bool, optional
        Whether to train the alpha parameter that blends the two networks (default is True).
    center_init_method : str, optional
        Method for initializing centers (default is 'random').
    center_distribution : str, optional
        Distribution for center initialization if 'random' is chosen (default is 'normal').
    center_uniform_range : tuple, optional
        Range for uniform distribution initialization (default is None).
    center_normal_var : float, optional
        Variance for normal distribution initialization (default is 1).

    Methods:
    --------
    trainable_params() -> int:
        Returns the total number of trainable parameters.
    __call__(input: np.ndarray) -> np.ndarray:
        Forward pass through the network.
    Adam_init():
        Initializes Adam optimizer parameters.
    update(...):
        Updates the network parameters using gradients and Adam or SGD optimizer.
    backward(...):
        Backpropagates the error and computes gradients for network parameters.
    """

    def __init__(self, input_size: int, output_size: int, batch_size: int = 32,
                 train_center: bool = True, train_var: bool = True, data=None,
                 var_init_method: str = 'average', var_init_const: float = 1.0,
                 train_alpha: bool = True, center_init_method: str = 'random',
                 center_distribution: str = 'normal',
                 center_uniform_range: tuple = None, center_normal_var: float = 1.0):

        """
        Initializes the RBF network with specified parameters and initializations.

        Parameters:
        -----------
        input_size : int
            Number of input features.
        output_size : int
            Number of output RBF units.
        batch_size : int, optional
            Number of samples in a batch (default is 32).
        train_center : bool, optional
            Whether to allow training of center weights (default is True).
        train_var : bool, optional
            Whether to allow training of variances (default is True).
        data : np.ndarray, optional
            Data used for K-means initialization if required (default is None).
        var_init_method : str, optional
            Method to initialize variances ('average', 'constant', or 'max', default is 'average').
        var_init_const : float, optional
            Constant value for variance initialization if 'constant' method is used (default is 1.0).
        train_alpha : bool, optional
            Whether to train the blend factor alpha (default is True).
        center_init_method : str, optional
            Method to initialize centers (default is 'random').
        center_distribution : str, optional
            Distribution for center initialization (default is 'normal').
        center_uniform_range : tuple, optional
            Range for uniform initialization (default is None).
        center_normal_var : float, optional
            Variance for normal initialization (default is 1.0).
        """
        
        # Store configuration and initialize centers with helper function
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.train_center = train_center
        self.train_var = train_var
        self.train_alpha = train_alpha
        
        # Initialize upper and lower network centers
        self.upper_center = RBF_weight_init(input_size, output_size, method=center_init_method,
                                            distribution=center_distribution, ranges=center_uniform_range,
                                            var=center_normal_var, data=data)
        self.lower_center = RBF_weight_init(input_size, output_size, method=center_init_method,
                                            distribution=center_distribution, ranges=center_uniform_range,
                                            var=center_normal_var, data=data)

        # Initialize alpha (blend factor) with a default value of 0.5
        self.alpha = np.full((output_size, 1), 0.5)

        # Initialize variances based on the chosen method
        if var_init_method == 'constant':
            self.upper_var = np.full((output_size, 1), var_init_const)
            self.lower_var = np.full((output_size, 1), var_init_const)
        elif var_init_method == 'average':
            self.upper_var = np.mean(self.upper_center, axis=1).reshape((-1, 1)) / output_size
            self.lower_var = np.mean(self.lower_center, axis=1).reshape((-1, 1)) / output_size
        elif var_init_method == 'max':
            self.upper_var = np.max(self.upper_center, axis=1).reshape((-1, 1)) / np.sqrt(2 * output_size)
            self.lower_var = np.max(self.lower_center, axis=1).reshape((-1, 1)) / np.sqrt(2 * output_size)

        # Prepare output storage arrays for forward pass
        self.upper_net = np.zeros((batch_size, output_size, 1))
        self.lower_net = np.zeros((batch_size, output_size, 1))
        self.minmax_reverse_stat = np.zeros((batch_size, output_size))
        self.final_output = np.zeros((batch_size, output_size, 1))
        self.upper_output = np.zeros((batch_size, output_size, 1))
        self.lower_output = np.zeros((batch_size, output_size, 1))

    #################################################################

    def trainable_params(self) -> int:
        """
        Calculates the total number of trainable parameters.

        Returns:
        --------
        int
            The total count of trainable parameters in the network.
        """
        params = 0
        if self.train_center:
            params += 2 * np.size(self.upper_center)  # Both upper and lower centers
        if self.train_var:
            params += 2 * np.size(self.upper_var)  # Both upper and lower variances
        if self.train_alpha:
            params += np.size(self.alpha)  # Alpha parameter
        return params

    #################################################################

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the RBF network.

        Parameters:
        -----------
        input : np.ndarray
            Input data array with shape (batch_size, input_size).

        Returns:
        --------
        np.ndarray
            The output of the RBF network with shape (batch_size, output_size).
        """
        input = input.reshape((-1, self.input_size))  # Ensure input is reshaped correctly
        self.input = input

        if self.batch_size < input.shape[0]:
            raise ValueError('Data batch size cannot be larger than model batch size')

        for batch_index, input_vector in enumerate(input):
            # Calculate distances between input vector and centers for both networks
            self.upper_net[batch_index] = np.linalg.norm((np.repeat(input_vector.reshape(1, -1), 
                                                                    self.output_size, axis=0) -
                                                          self.upper_center), axis=1).reshape((-1, 1))
            self.lower_net[batch_index] = np.linalg.norm((np.repeat(input_vector.reshape(1, -1),
                                                                    self.output_size, axis=0) -
                                                          self.lower_center), axis=1).reshape((-1, 1))
            
            # Compute RBF output for upper and lower networks with numerical stability handling
            try:
                self.upper_output[batch_index] = np.exp(-0.5 * np.square(
                    self.upper_net[batch_index].ravel() / self.upper_var.ravel())).reshape(-1, 1)
                self.lower_output[batch_index] = np.exp(-0.5 * np.square(
                    self.lower_net[batch_index].ravel() / self.lower_var.ravel())).reshape(-1, 1)
            except FloatingPointError:
                self.upper_output[batch_index] = np.exp(-0.5 * np.square(
                    self.upper_net[batch_index].ravel() / (self.upper_var.ravel() + 1e-7))).reshape(-1, 1)
                self.lower_output[batch_index] = np.exp(-0.5 * np.square(
                    self.lower_net[batch_index].ravel() / (self.lower_var.ravel() + 1e-7))).reshape(-1, 1)

            # Combine upper and lower outputs and compute the final output using alpha
            up_out = self.upper_output[batch_index]
            low_out = self.lower_output[batch_index]
            concat_out = np.concatenate((up_out, low_out), axis=1)
            self.minmax_reverse_stat[batch_index] = np.argmax(concat_out, axis=1)

            self.upper_output[batch_index] = np.max(concat_out, axis=1).reshape((-1, 1))
            self.lower_output[batch_index] = np.min(concat_out, axis=1).reshape((-1, 1))
            self.final_output[batch_index] = self.alpha * self.upper_output[batch_index] + \
                                             (1 - self.alpha) * self.lower_output[batch_index]

        return self.final_output[:input.shape[0]].reshape((-1, self.output_size))

    #################################################################

    def Adam_init(self) -> None:
        """
        Initializes Adam optimizer variables for center, variance, and alpha parameters if they are trainable.

        Initializes the following attributes:
            - self.t: Timestep counter for Adam optimizer.
            - self.upper_center_mt, self.lower_center_mt, self.upper_center_vt, self.lower_center_vt: First (mt) and 
              second (vt) moment estimates for upper and lower centers.
            - self.upper_var_mt, self.lower_var_mt, self.upper_var_vt, self.lower_var_vt: Moment estimates for upper 
              and lower variances if biases are used.
            - self.alpha_mt, self.alpha_vt: Moment estimates for alpha parameter if it is trainable.
        """
        self.t = 0  # Initialize timestep for Adam optimizer

        if self.train_center:
            # Initialize first (mt) and second (vt) moments for center parameters
            self.upper_center_mt = np.zeros(self.upper_center.shape)
            self.lower_center_mt = np.zeros(self.lower_center.shape)
            self.upper_center_vt = np.zeros(self.upper_center.shape)
            self.lower_center_vt = np.zeros(self.lower_center.shape)

        if self.train_var:
            # Initialize first and second moments for variance parameters
            self.upper_var_mt = np.zeros(self.upper_var.shape)
            self.lower_var_mt = np.zeros(self.lower_var.shape)
            self.upper_var_vt = np.zeros(self.upper_var.shape)
            self.lower_var_vt = np.zeros(self.lower_var.shape)

        if self.train_alpha:
            # Initialize first and second moments for alpha parameters
            self.alpha_mt = np.zeros(self.alpha.shape)
            self.alpha_vt = np.zeros(self.alpha.shape)

    #################################################################

    def update(self, grad_c_up: np.ndarray, grad_c_low: np.ndarray, grad_var_up: np.ndarray, grad_var_low: np.ndarray,
               grad_alpha: np.ndarray, method: str = 'Adam', learning_rate: float = 1e-3,
               bias_learning_rate: float = 2e-4, adam_beta1: float = 0.9, adam_beta2: float = 0.99) -> None:
        """
        Updates the model parameters using specified optimization method, either Adam or SGD.
        
        Parameters:
        -----------
        grad_c_up, grad_c_low : np.ndarray
            Gradients for the upper and lower centers, respectively.
        
        grad_var_up, grad_var_low : np.ndarray
            Gradients for the upper and lower variances, respectively.
        
        grad_alpha : np.ndarray
            Gradient for the alpha parameter.

        method : str, default='Adam'
            Optimization method. Supported values are 'Adam' and 'SGD'.
        
        learning_rate : float, default=1e-3
            Learning rate for center and alpha updates.
        
        bias_learning_rate : float, default=2e-4
            Learning rate for variance updates (if biases are enabled).
        
        adam_beta1, adam_beta2 : float, default=(0.9, 0.99)
            Adam optimizer hyperparameters for first and second moment decay rates.

        Returns:
        --------
        None
        """
        eps = 1e-7  # Small constant to prevent division by zero in Adam

        if method == 'Adam':
            self.t += 1  # Increment timestep

            if self.train_center:
                # Update moments for center parameters
                self.upper_center_mt = adam_beta1 * self.upper_center_mt + (1 - adam_beta1) * grad_c_up
                self.upper_center_vt = adam_beta2 * self.upper_center_vt + (1 - adam_beta2) * np.square(grad_c_up)
                
                self.lower_center_mt = adam_beta1 * self.lower_center_mt + (1 - adam_beta1) * grad_c_low
                self.lower_center_vt = adam_beta2 * self.lower_center_vt + (1 - adam_beta2) * np.square(grad_c_low)
                
                # Compute bias-corrected estimates for center parameters
                m_hat_c_up = self.upper_center_mt / (1 - adam_beta1 ** self.t)
                v_hat_c_up = self.upper_center_vt / (1 - adam_beta2 ** self.t)
                
                m_hat_c_low = self.lower_center_mt / (1 - adam_beta1 ** self.t)
                v_hat_c_low = self.lower_center_vt / (1 - adam_beta2 ** self.t)
                
                # Update center parameters
                delta_c_up = learning_rate * m_hat_c_up / (np.sqrt(v_hat_c_up) + eps)
                delta_c_low = learning_rate * m_hat_c_low / (np.sqrt(v_hat_c_low) + eps)

            if self.train_var:
                # Update moments for variance parameters
                self.upper_var_mt = adam_beta1 * self.upper_var_mt + (1 - adam_beta1) * grad_var_up
                self.upper_var_vt = adam_beta2 * self.upper_var_vt + (1 - adam_beta2) * np.square(grad_var_up)
                
                self.lower_var_mt = adam_beta1 * self.lower_var_mt + (1 - adam_beta1) * grad_var_low
                self.lower_var_vt = adam_beta2 * self.lower_var_vt + (1 - adam_beta2) * np.square(grad_var_low)
                
                # Compute bias-corrected estimates for variance parameters
                m_hat_var_up = self.upper_var_mt / (1 - adam_beta1 ** self.t)
                v_hat_var_up = self.upper_var_vt / (1 - adam_beta2 ** self.t)
                
                m_hat_var_low = self.lower_var_mt / (1 - adam_beta1 ** self.t)
                v_hat_var_low = self.lower_var_vt / (1 - adam_beta2 ** self.t)
                
                # Update variance parameters
                delta_var_up = bias_learning_rate * m_hat_var_up / (np.sqrt(v_hat_var_up) + eps)
                delta_var_low = bias_learning_rate * m_hat_var_low / (np.sqrt(v_hat_var_low) + eps)

            if self.train_alpha:
                # Update moments and compute bias-corrected estimates for alpha
                self.alpha_mt = adam_beta1 * self.alpha_mt + (1 - adam_beta1) * grad_alpha
                self.alpha_vt = adam_beta2 * self.alpha_vt + (1 - adam_beta2) * np.square(grad_alpha)
                
                # Update alpha parameter
                delta_alpha = learning_rate * self.alpha_mt / (np.sqrt(self.alpha_vt) + eps)
        else:
            # Simple SGD updates for center, variance, and alpha
            if self.train_center:
                delta_c_up = learning_rate * grad_c_up
                delta_c_low = learning_rate * grad_c_low
            if self.train_var:
                delta_var_up = bias_learning_rate * grad_var_up
                delta_var_low = bias_learning_rate * grad_var_low
            if self.train_alpha:
                delta_alpha = learning_rate * grad_alpha

        # Apply parameter updates if the corresponding flags are set
        if self.train_center:
            self.upper_center -= delta_c_up
            self.lower_center -= delta_c_low
        if self.train_var:
            self.upper_var -= delta_var_up
            self.lower_var -= delta_var_low
        if self.train_alpha:
            self.alpha -= delta_alpha

    #################################################################

    def backward(self, error_batch: np.ndarray, method: str = 'Adam', 
                 learning_rate: float = 1e-3, bias_learning_rate: float = 2e-4, 
                 adam_beta1: float = 0.9, adam_beta2: float = 0.99) -> np.ndarray:
        """
        Executes the backward pass to compute gradients for centers, variances, and alpha parameter 
        based on error propagation through the network.

        Parameters:
        -----------
        error_batch : np.ndarray
            Error values for each sample in the batch.

        method : str, default='Adam'
            Optimization method, either 'Adam' or 'SGD'.
        
        learning_rate : float, default=1e-3
            Learning rate for updating center and alpha parameters.
        
        bias_learning_rate : float, default=2e-4
            Learning rate for variance updates.
        
        adam_beta1 : float, default=0.9
            Decay rate for the first moment estimate in Adam optimizer.
        
        adam_beta2 : float, default=0.99
            Decay rate for the second moment estimate in Adam optimizer.

        Returns:
        --------
        error_out : np.ndarray
            Error output array based on computed gradients.
        """
        error_out = np.zeros(self.input.shape)  # Initialize error output array with shape of input

        grad_c_up = None
        grad_c_low = None
        if self.train_center:
            # Initialize gradients for upper and lower centers if centers are trainable
            grad_c_up = np.zeros(self.upper_center.shape)
            grad_c_low = np.zeros(self.lower_center.shape)
        
        grad_var_up = None
        grad_var_low = None
        if self.train_var:
            # Initialize gradients for upper and lower variances if variances are trainable
            grad_var_up = np.zeros(self.upper_var.shape)
            grad_var_low = np.zeros(self.lower_var.shape)

        grad_alpha = None
        if self.train_alpha:
            # Initialize gradient for alpha if alpha is trainable
            grad_alpha = np.zeros(self.alpha.shape)

        # Iterate over each batch index to compute gradients based on batch errors
        for batch_index, one_batch_error in enumerate(error_batch):
            if self.train_alpha:
                # Compute gradient for alpha using the difference between upper and lower outputs
                grad_alpha += one_batch_error.reshape((-1, 1)) * \
                    (self.upper_output[batch_index] - self.lower_output[batch_index])

            e_max = self.alpha * one_batch_error.reshape((-1, 1))  # Compute maximum error component
            e_min = (1 - self.alpha) * one_batch_error.reshape((-1, 1))  # Compute minimum error component

            # Separate error into upper and lower errors based on minmax reverse state
            e_upper = e_max * np.logical_not(self.minmax_reverse_stat[batch_index].reshape((-1, 1))) + \
                      e_min * self.minmax_reverse_stat[batch_index].reshape((-1, 1))
            e_lower = e_min * np.logical_not(self.minmax_reverse_stat[batch_index].reshape((-1, 1))) + \
                      e_max * self.minmax_reverse_stat[batch_index].reshape((-1, 1))

            if self.train_center:
                # Calculate gradient for upper centers
                grad_c_up = np.diag(e_upper.ravel() * self.upper_output[batch_index].ravel() * \
                                    self.upper_var.ravel() ** -2) @\
                            (np.repeat(self.input[batch_index].reshape(1, -1), \
                                       self.output_size, axis=0) - self.upper_center)
                # Calculate gradient for lower centers
                grad_c_low = np.diag(e_lower.ravel() * self.lower_output[batch_index].ravel() * \
                                    self.lower_var.ravel() ** -2) @\
                            (np.repeat(self.input[batch_index].reshape(1, -1), \
                                       self.output_size, axis=0) - self.lower_center)
                
            if self.train_var:
                # Calculate gradient for upper variances
                grad_var_up += (e_upper.ravel() * self.upper_net[batch_index].ravel() ** 2 *\
                    self.upper_var.ravel() ** -3 * self.upper_output[batch_index].ravel()).reshape((-1, 1))
                # Calculate gradient for lower variances
                grad_var_low += (e_lower.ravel() * self.lower_net[batch_index].ravel() ** 2 *\
                    self.lower_var.ravel() ** -3 * self.lower_output[batch_index].ravel()).reshape((-1, 1))
            

            # Calculate error contribution for upper centers based on input and center deviation
            error_x_up = (e_upper.ravel() * self.upper_output[batch_index].ravel() * \
                       self.upper_var.ravel() ** -2).reshape((-1, 1))
            e_x_up = (np.ones((self.input_size, self.output_size)) @ error_x_up).ravel() *\
                (np.sum(2 * self.upper_center, axis=0).ravel() - 4 * self.input[batch_index].ravel())
            
            # Calculate error contribution for lower centers based on input and center deviation
            error_x_low = (e_lower.ravel() * self.lower_output[batch_index].ravel() * \
                       self.lower_var.ravel() ** -2).reshape((-1, 1))
            e_x_low = (np.ones((self.input_size, self.output_size)) @ error_x_low).ravel() *\
                (np.sum(2 * self.lower_center, axis=0).ravel() - 4 * self.input[batch_index].ravel())
            
            error_out[batch_index] = e_x_low + e_x_up  # Summing error contributions for each batch

        if self.train_center:
            # Normalize center gradients by batch size if centers are trainable
            grad_c_up /= error_batch.shape[0]
            grad_c_low /= error_batch.shape[0]

        if self.train_var:
            # Normalize variance gradients by batch size if variances are trainable
            grad_var_up /= error_batch.shape[0]
            grad_var_low /= error_batch.shape[0]

        if self.train_alpha:
            # Normalize alpha gradient by batch size if alpha is trainable
            grad_alpha /= error_batch.shape[0]

        # Update model parameters using computed gradients
        self.update(grad_c_up, grad_c_low, grad_var_up, grad_var_low, grad_alpha,
                    method=method, learning_rate=learning_rate,
                    bias_learning_rate=bias_learning_rate, adam_beta1=adam_beta1, adam_beta2=adam_beta2)

        return error_out  # Return error output based on gradients computed during backward pass
