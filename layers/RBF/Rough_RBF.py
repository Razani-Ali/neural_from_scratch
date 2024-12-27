import numpy as np
from initializers.weight_initializer import RBF_init
from optimizers.set_optimizer import init_optimizer


class RoughRBF:
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
    train_blending : bool, optional
        Whether to train the alpha parameter that blends the two networks (default is True).

    Methods:
    --------
    trainable_params() -> int:
        Returns the total number of trainable parameters.
    __call__(input: np.ndarray) -> np.ndarray:
        Forward pass through the network.
    optimizer_init():
        Initializes optimizer parameters for current layer.
    update(...):
        Updates the network parameters using gradients and Adam or SGD optimizer.
    backward(...):
        Backpropagates the error and computes gradients for network parameters.
    """

    def __init__(self, input_size: int, output_size: int, batch_size: int = 32,
                 train_center: bool = True, train_var: bool = True, data=None,
                 train_blending: bool = True, 
                 center_init_method: str = 'random',
                 center_distribution: str = 'uniform',
                 center_uniform_range: tuple = (-1, 1), 
                 center_normal_var: float = 1.0,
                 var_init_method: str = 'zeros', 
                 var_constant: float = 1.0,
                 lower_center_division_factor: float = 1.0,
                 lower_var_division_factor: float = 1.0):

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
        center_init_method : str, optional
            Method to initialize centers (default is 'random').
        center_distribution : str, optional
            Distribution for center initialization (default is 'normal').
        center_uniform_range : tuple, optional
            Range for uniform initialization (default is None).
        center_normal_var : float, optional
            Variance for normal initialization (default is 1.0).
        var_init_method : str, optional
            Method for variance initialization (default is 'zeros').
        var_constant : float, optional
            Constant value for variance initialization (default is 1.0).
        lower_center_division_factor : float, optional
            Factor to divide upper centers to get lower centers (default is 2.0).
        lower_var_division_factor : float, optional
            Factor to divide upper variances to get lower variances (default is 2.0).
        """
        # Store configuration and initialize centers with helper function
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.train_center = train_center
        self.train_var = train_var
        self.train_blending = train_blending
        self.activation = 'Gaussian Kernel'

        # Initialize upper network centers and variances
        self.upper_center, upper_var = RBF_init(
            input_size, output_size, method=center_init_method,
            distribution=center_distribution, ranges=center_uniform_range,
            var=center_normal_var, data=data, var_init_method=var_init_method, 
            var_constant=var_constant
        )
        self.upper_var = upper_var.ravel()

        # Initialize lower network centers and variances based on upper network
        self.lower_center = self.upper_center / lower_center_division_factor
        self.lower_var = self.upper_var / lower_var_division_factor

        # Initialize alpha (blend factor) with a default value of 0.5
        self.blending_factor = np.full((output_size, ), 0.5)

        # Prepare output storage arrays for forward pass
        self.upper_net = np.zeros((batch_size, output_size))
        self.lower_net = np.zeros((batch_size, output_size))
        self.minmax_reverse_stat = np.zeros((batch_size, output_size))
        self.final_output = np.zeros((batch_size, output_size))
        self.upper_output = np.zeros((batch_size, output_size))
        self.lower_output = np.zeros((batch_size, output_size))

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
        if self.train_blending:
            params += np.size(self.blending_factor)  # Alpha parameter
        return params
    
    #################################################################

    def all_params(self) -> int:
        """
        Calculates the total number of parameters.

        Returns:
        --------
        int
            The total count of parameters in the network.
        """
        return 2 * np.size(self.upper_center) + 2 * np.size(self.upper_var) + np.size(self.blending_factor)

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
        self.input = input

        if self.batch_size < input.shape[0]:
            raise ValueError('Data batch size cannot be larger than model batch size')

        for batch_index, input_vector in enumerate(input):
            input_vector = input_vector.reshape(1, -1)
            # Calculate distances between input vector and centers for both networks
            self.upper_net[batch_index] = np.linalg.norm((np.repeat(input_vector, 
                                                                    self.output_size, axis=0) -
                                                          self.upper_center), axis=1).ravel()
            self.lower_net[batch_index] = np.linalg.norm((np.repeat(input_vector,
                                                                    self.output_size, axis=0) -
                                                          self.lower_center), axis=1).ravel()
            
            # Compute RBF output for upper and lower networks with numerical stability handling
            try:
                self.upper_output[batch_index] = np.exp(-0.5 * np.square(
                    self.upper_net[batch_index] / self.upper_var))
                self.lower_output[batch_index] = np.exp(-0.5 * np.square(
                    self.lower_net[batch_index] / self.lower_var))
            except FloatingPointError:
                self.upper_output[batch_index] = np.exp(-0.5 * np.square(
                    self.upper_net[batch_index] / (self.upper_var + 1e-7)))
                self.lower_output[batch_index] = np.exp(-0.5 * np.square(
                    self.lower_net[batch_index] / (self.lower_var + 1e-7)))

            # Combine upper and lower outputs and compute the final output using alpha
            up_out = self.upper_output[batch_index].reshape((-1,1))
            low_out = self.lower_output[batch_index].reshape((-1,1))
            concat_out = np.concatenate((up_out, low_out), axis=1)
            self.minmax_reverse_stat[batch_index] = np.argmax(concat_out, axis=1)

            self.upper_output[batch_index] = np.max(concat_out, axis=1)
            self.lower_output[batch_index] = np.min(concat_out, axis=1)
            self.final_output[batch_index] = self.blending_factor * self.upper_output[batch_index] + \
                                             (1 - self.blending_factor) * self.lower_output[batch_index]
        
        # Return the final output for all input samples
        batch_index += 1
        return self.final_output[:batch_index]

    #################################################################
    
    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initialize the optimizer for the Rough RBF model using specified method.

        Parameters:
        ----------
        optimizer : str, optional
            Name of the optimizer to be used (default is 'Adam').
        **kwargs : dict
            Additional parameters for optimizer configuration.

        Returns:
        -------
        None
        """
        # Calls a function to initialize the optimizer by passing trainable parameters and method
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, grads: np.ndarray, learning_rate: float = 1e-3) -> None:
        """
        Update model parameters using computed gradients and specified learning rate.

        Parameters:
        ----------
        grads : np.ndarray
            Gradients for model parameters.
        learning_rate : float, optional
            Learning rate for parameter updates (default is 0.001).

        Returns:
        -------
        None
        """
        # Compute delta updates by applying the optimizer on gradients and learning rate
        deltas = self.Optimizer(grads, learning_rate)
        ind2 = 0  # Initialize index pointer to start processing parameter deltas

        # Update centers if trainable
        if self.train_center:
            # Slice and reshape deltas for the upper centers, then subtract from current centers
            ind1 = ind2
            ind2 += int(np.size(self.upper_center))
            delta_cen = deltas[ind1:ind2].reshape(self.upper_center.shape)
            self.upper_center -= delta_cen

            # Slice and reshape deltas for the lower centers, then subtract from current centers
            ind1 = ind2
            ind2 += int(np.size(self.lower_center))
            delta_cen = deltas[ind1:ind2].reshape(self.lower_center.shape)
            self.lower_center -= delta_cen

        # Update variances if trainable
        if self.train_var:
            # Slice and reshape deltas for upper variances, then subtract from current variances
            ind1 = ind2
            ind2 += np.size(self.upper_var)
            delta_var = deltas[ind1:ind2].reshape(self.upper_var.shape)
            self.upper_var -= delta_var

            # Slice and reshape deltas for lower variances, then subtract from current variances
            ind1 = ind2
            ind2 += np.size(self.lower_var)
            delta_var = deltas[ind1:ind2].reshape(self.lower_var.shape)
            self.lower_var -= delta_var

        # Update blending factor if trainable
        if self.train_blending:
            # Slice and reshape deltas for blending factor, then subtract from current blending factor
            ind1 = ind2
            ind2 += np.size(self.blending_factor)
            delta_blend = deltas[ind1:ind2].reshape(self.blending_factor.shape)
            self.blending_factor -= delta_blend

    #################################################################

    def backward(self, 
                 error_batch: np.ndarray, 
                 learning_rate: float = 1e-3, 
                 return_error: bool = False, 
                 return_grads: bool = False, 
                 modify: bool = True
                 ):
        """
        Perform backward propagation to calculate gradients and update parameters.

        Parameters:
        ----------
        error_batch : np.ndarray
            Batch of errors for each output node.
        learning_rate : float, optional
            Learning rate for updating parameters (default is 0.001).
        return_error : bool, optional
            If True, returns error propagated to previous layer (default is False).
        return_grads : bool, optional
            If True, returns computed gradients (default is False).
        modify : bool, optional
            If True, updates parameters using gradients (default is True).

        Returns:
        -------
        Optional[Union[Dict[str, Any], np.ndarray]]
            Returns propagated error and/or gradients if specified, otherwise None.
        """
        # Initialize error to propagate to the previous layer if required
        if return_error:
            error_in = np.zeros(self.input.shape)

        # Initialize gradient arrays for centers, variances, and blending factor if they are trainable
        grad_c_up = np.zeros(self.upper_center.shape) if self.train_center else None
        grad_c_low = np.zeros(self.lower_center.shape) if self.train_center else None
        grad_var_up = np.zeros(self.upper_var.shape) if self.train_var else None
        grad_var_low = np.zeros(self.lower_var.shape) if self.train_var else None
        grad_alpha = np.zeros(self.blending_factor.shape) if self.train_blending else None

        # Loop through each batch to compute gradients based on error for each output node
        for batch_index, one_batch_error in enumerate(error_batch):

            # Calculate gradient for blending factor (alpha) based on the difference in upper and lower outputs
            if self.train_blending:
                grad_alpha += one_batch_error * (self.upper_output[batch_index] - self.lower_output[batch_index])

            # Compute scaled error components using blending factor
            e_max = self.blending_factor * one_batch_error
            e_min = (1 - self.blending_factor) * one_batch_error

            # Distribute error to upper and lower layers using minmax reverse state
            e_upper = e_max * np.logical_not(self.minmax_reverse_stat[batch_index]) + e_min *\
                self.minmax_reverse_stat[batch_index]
            e_lower = e_min * np.logical_not(self.minmax_reverse_stat[batch_index]) + e_max *\
                self.minmax_reverse_stat[batch_index]

            # Compute gradients for centers
            if self.train_center:
                grad_c_up += np.diag(e_upper * self.upper_output[batch_index] * self.upper_var ** -2) @ \
                             (np.repeat(self.input[batch_index].reshape(1, -1), self.output_size, axis=0) - self.upper_center)
                grad_c_low += np.diag(e_lower * self.lower_output[batch_index] * self.lower_var ** -2) @ \
                              (np.repeat(self.input[batch_index].reshape(1, -1), self.output_size, axis=0) - self.lower_center)

            # Compute gradients for variances
            if self.train_var:
                grad_var_up += (e_upper * self.upper_net[batch_index] ** 2 * self.upper_var ** -3 *\
                                self.upper_output[batch_index])
                grad_var_low += (e_lower * self.lower_net[batch_index] ** 2 * self.lower_var ** -3 *\
                                 self.lower_output[batch_index])

            # Compute error contributions for input based on centers
            if return_error:
                error_x_up = (e_upper * self.upper_output[batch_index] * self.upper_var ** -2).reshape((-1, 1))
                e_x_up = (np.ones((self.input_size, self.output_size)) @ error_x_up).ravel() *\
                    (np.sum(2 * self.upper_center, axis=0).ravel() - 4 * self.input[batch_index])

                error_x_low = (e_lower * self.lower_output[batch_index] * self.lower_var ** -2).reshape((-1, 1))
                e_x_low = (np.ones((self.input_size, self.output_size)) @ error_x_low) *\
                    (np.sum(2 * self.lower_center, axis=0).ravel() - 4 * self.input[batch_index])

                # Sum contributions for each batch index
                error_in[batch_index] = e_x_low + e_x_up

        # Normalize gradients by batch size if trainable
        if self.train_center:

            grad_c_up /= error_batch.shape[0]
            grad_c_low /= error_batch.shape[0]

        if self.train_var:

            grad_var_up /= error_batch.shape[0]
            grad_var_low /= error_batch.shape[0]

        if self.train_blending:
            
            grad_alpha /= error_batch.shape[0]

        # Concatenate gradients and update if required
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
        if grads is not None:
            if grad_c_up is not None:
                grads = np.concatenate((grads, grad_c_up.reshape((-1, 1))))
                grads = np.concatenate((grads, grad_c_low.reshape((-1, 1))))
            if grad_var_up is not None:
                grads = np.concatenate((grads, grad_var_up.reshape((-1, 1))))
                grads = np.concatenate((grads, grad_var_low.reshape((-1, 1))))
            if grad_alpha is not None:
                grads = np.concatenate((grads, grad_alpha.reshape((-1, 1))))
        if modify:
                        # Update model parameters using the computed gradients if modify flag is set to True
            self.update(grads, learning_rate=learning_rate)

        # Return output error, gradients, or both if specified by parameters
        if return_error and return_grads:
            # Return a dictionary with propagated error and computed gradients
            return {'error_in': error_in, 'gradients': grads}
        elif return_error:
            # Return only the propagated error if specified
            return error_in
        elif return_grads:
            # Return only the gradients if specified
            return grads

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class TimeRoughRBF:
    """
    A class representing a rough Radial Basis Function (RBF) network 
    with dual-layer structure (upper and lower networks) and 
    an optional adaptive blend factor (alpha).

    Attributes:
    -----------
    time_steps : int
        Number of time steps or sequences.
    input_size : int
        The number of input features to the RBF network.
    output_size : int
        The number of output neurons (RBF units).
    batch_size : int
        The number of samples in each batch.
    train_center : bool
        Whether to train the centers of the RBFs.
    train_var : bool
        Whether to train the variances of the RBFs.
    train_blending : bool
        Whether to train the alpha parameter that blends the two networks.
    activation : str
        Activation function used (fixed to 'Gaussian Kernel').
    upper_center : np.ndarray
        Centers of the upper network.
    lower_center : np.ndarray
        Centers of the lower network.
    upper_var : np.ndarray
        Variances of the upper RBF neurons.
    lower_var : np.ndarray
        Variances of the lower RBF neurons.
    blending_factor : np.ndarray
        Alpha parameter blending the outputs of upper and lower networks.

    Methods:
    --------
    trainable_params() -> int:
        Returns the total number of trainable parameters.
    all_params() -> int:
        Returns the total number of parameters in the network.
    __call__(batch_index: int, seq_index: int, input: np.ndarray) -> np.ndarray:
        Performs a forward pass through the layer.
    optimizer_init(optimizer: str = 'Adam', **kwargs) -> None:
        Initializes optimizer parameters for the current layer.
    update(batch_size: int, learning_rate: float, grads: Optional[np.ndarray]) -> None:
        Updates the network parameters using gradients.
    return_grads() -> Optional[np.ndarray]:
        Returns the computed gradients for trainable parameters.
    backward(batch_index: int, seq_index: int, error: np.ndarray) -> np.ndarray:
        Backpropagates the error and computes gradients for network parameters.
    """

    def __init__(self, 
                 time_steps: int, 
                 input_size: int, 
                 output_size: int, 
                 batch_size: int = 32, 
                 train_center: bool = True, 
                 train_var: bool = True, 
                 data: np.ndarray = None, 
                 var_init_method: str = 'max', 
                 var_init_const: float = 1.0, 
                 train_blending: bool = True, 
                 center_init_method: str = 'random', 
                 center_distribution: str = 'normal', 
                 center_uniform_range: tuple = None, 
                 center_normal_var: float = 1.0, 
                 lower_center_div_factor: float = 1.0, 
                 lower_var_div_factor: float = 1.0):
        """
        Initializes the RBF network with specified parameters and initializations.

        Parameters:
        -----------
        time_steps : int
            Number of time steps or sequences.
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
            Method to initialize variances ('average', 'constant', or 'max', default is 'max').
        var_init_const : float, optional
            Constant value for variance initialization if 'constant' method is used (default is 1.0).
        train_blending : bool, optional
            Whether to train the blend factor alpha (default is True).
        center_init_method : str, optional
            Method to initialize centers (default is 'random').
        center_distribution : str, optional
            Distribution for center initialization (default is 'normal').
        center_uniform_range : tuple, optional
            Range for uniform initialization (default is None).
        center_normal_var : float, optional
            Variance for normal initialization (default is 1.0).
        lower_center_div_factor : float, optional
            Factor to divide the upper centers to initialize the lower centers (default is 2.0).
        lower_var_div_factor : float, optional
            Factor to divide the upper variances to initialize the lower variances (default is 1.5).
        """
        # Configuration
        self.time_steps = time_steps
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.train_center = train_center
        self.train_var = train_var
        self.train_blending = train_blending
        self.activation = 'Gaussian Kernel'

        # Initialize upper centers and variances
        self.upper_center, upper_var = RBF_init(input_size, output_size, 
                                                     method=center_init_method,
                                                     distribution=center_distribution, 
                                                     ranges=center_uniform_range, 
                                                     var=center_normal_var, 
                                                     data=data, 
                                                     var_init_method=var_init_method, 
                                                     var_constant=var_init_const)
        self.upper_var = upper_var.ravel()

        # Initialize lower centers and variances
        self.lower_center = self.upper_center / lower_center_div_factor
        self.lower_var = self.upper_var / lower_var_div_factor

        # Initialize blending factor
        self.blending_factor = np.full((output_size, 1), 0.5)

        # Prepare intermediate storage arrays
        self.upper_net = np.zeros((batch_size, time_steps, output_size, 1))
        self.lower_net = np.zeros((batch_size, time_steps, output_size, 1))
        self.minmax_reverse_stat = np.zeros((batch_size, time_steps, output_size))
        self.output = np.zeros((batch_size, time_steps, output_size, 1))
        self.input = np.zeros((batch_size, time_steps, input_size, 1))
        self.upper_output = np.zeros((batch_size, time_steps, output_size, 1))
        self.lower_output = np.zeros((batch_size, time_steps, output_size, 1))

        # Gradients
        self.grad_cen_up = np.zeros(self.upper_center.shape) if self.train_center else None
        self.grad_cen_low = np.zeros(self.lower_center.shape) if self.train_center else None
        self.grad_var_up = np.zeros(self.upper_var.shape) if self.train_var else None
        self.grad_var_low = np.zeros(self.lower_var.shape) if self.train_var else None
        self.grad_blend = np.zeros(self.blending_factor.shape) if self.train_blending else None

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
        if self.train_blending:
            params += np.size(self.blending_factor)  # Alpha parameter
        return params

    #################################################################

    def all_params(self) -> int:
        """
        Calculates the total number of parameters.

        Returns:
        --------
        int
            The total count of parameters in the network.
        """
        return 2 * np.size(self.upper_center) + 2 * np.size(self.upper_var) + np.size(self.blending_factor)

    #################################################################

    def __call__(self, batch_index: int, seq_index: int, input: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the layer.

        Parameters:
        ----------
        batch_index : int
            Current batch index.
        seq_index : int
            Current sequence index.
        input : np.ndarray
            Input data array of shape (input_size,).

        Returns:
        -------
        np.ndarray
            Output data array of shape (output_size, 1).
        """
        self.input[batch_index, seq_index] = input.reshape((-1, 1))

        # Calculate distances between input vector and centers for both networks
        self.upper_net[batch_index, seq_index] = np.linalg.norm(
            (np.repeat(input.reshape(1, -1), self.output_size, axis=0) - self.upper_center),
            axis=1
        ).reshape((-1, 1))

        self.lower_net[batch_index, seq_index] = np.linalg.norm(
            (np.repeat(input.reshape(1, -1), self.output_size, axis=0) - self.lower_center),
            axis=1
        ).reshape((-1, 1))

        # Compute RBF output for upper and lower networks with numerical stability handling
        try:
            self.upper_output[batch_index, seq_index] = np.exp(-0.5 * np.square(
                self.upper_net[batch_index, seq_index].ravel() / self.upper_var.ravel())).reshape(-1, 1)
            self.lower_output[batch_index, seq_index] = np.exp(-0.5 * np.square(
                self.lower_net[batch_index, seq_index].ravel() / self.lower_var.ravel())).reshape(-1, 1)
        except FloatingPointError:
            self.upper_output[batch_index, seq_index] = np.exp(-0.5 * np.square(
                self.upper_net[batch_index, seq_index].ravel() / (self.upper_var.ravel() + 1e-7))).reshape(-1, 1)
            self.lower_output[batch_index, seq_index] = np.exp(-0.5 * np.square(
                self.lower_net[batch_index, seq_index].ravel() / (self.lower_var.ravel() + 1e-7))).reshape(-1, 1)

        # Combine upper and lower outputs and compute the final output using blending factor (alpha)
        up_out = self.upper_output[batch_index, seq_index]
        low_out = self.lower_output[batch_index, seq_index]
        concat_out = np.concatenate((up_out, low_out), axis=1)
        self.minmax_reverse_stat[batch_index, seq_index] = np.argmax(concat_out, axis=1)

        self.upper_output[batch_index, seq_index] = np.max(concat_out, axis=1).reshape((-1, 1))
        self.lower_output[batch_index, seq_index] = np.min(concat_out, axis=1).reshape((-1, 1))
        self.output[batch_index, seq_index] = self.blending_factor * self.upper_output[batch_index, seq_index] + \
                                              (1 - self.blending_factor) * self.lower_output[batch_index, seq_index]

        return self.output[batch_index, seq_index]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initialize the optimizer for the Rough RBF model using specified method.

        Parameters:
        ----------
        optimizer : str, optional
            Name of the optimizer to be used (default is 'Adam').
        **kwargs : dict
            Additional parameters for optimizer configuration.

        Returns:
        -------
        None
        """
        # Calls a function to initialize the optimizer by passing trainable parameters and method
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, batch_size: int, learning_rate: float = 1e-3, grads: np.ndarray = None) -> None:
        """
        Updates the layer's parameters based on calculated gradients.

        Parameters:
        -----------
        batch_size : int
            The size of the batch used for averaging gradients.
        learning_rate : float, optional
            The step size for parameter updates (default is 1e-3).
        grads : np.ndarray, optional
            Gradients with respect to trainable parameters.

        Returns:
        --------
        None
        """
        if grads is None:
            grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
            if grads is not None:
                if self.grad_cen_up is not None:
                    grads = np.concatenate((grads, self.grad_cen_up.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_cen_low.reshape((-1, 1))))
                if self.grad_var_up is not None:
                    grads = np.concatenate((grads, self.grad_var_up.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_var_low.reshape((-1, 1))))
                if self.grad_blend is not None:
                    grads = np.concatenate((grads, self.grad_blend.reshape((-1, 1))))
            grads /= batch_size

        # Compute deltas for parameters using the optimizer
        deltas = self.Optimizer(grads, learning_rate)
        ind2 = 0  # Pointer to track parameter indices

        # Update centers if trainable
        if self.train_center:
            ind1 = ind2
            ind2 += int(np.size(self.upper_center))
            delta_cen = deltas[ind1:ind2].reshape(self.upper_center.shape)
            self.upper_center -= delta_cen

            ind1 = ind2
            ind2 += int(np.size(self.lower_center))
            delta_cen = deltas[ind1:ind2].reshape(self.lower_center.shape)
            self.lower_center -= delta_cen

        # Update variances if trainable
        if self.train_var:
            ind1 = ind2
            ind2 += np.size(self.upper_var)
            delta_var = deltas[ind1:ind2].reshape(self.upper_var.shape)
            self.upper_var -= delta_var

            ind1 = ind2
            ind2 += np.size(self.lower_var)
            delta_var = deltas[ind1:ind2].reshape(self.lower_var.shape)
            self.lower_var -= delta_var

        # Update blending factor if trainable
        if self.train_blending:
            ind1 = ind2
            ind2 += np.size(self.blending_factor)
            delta_blend = deltas[ind1:ind2].reshape(self.blending_factor.shape)
            self.blending_factor -= delta_blend

        # Reset gradients
        self.grad_cen_up = self.grad_cen_up * 0 if self.train_center else None
        self.grad_cen_low = self.grad_cen_low * 0 if self.train_center else None
        self.grad_var_up = self.grad_var_up * 0 if self.train_var else None
        self.grad_var_low = self.grad_var_low * 0 if self.train_var else None
        self.grad_blend = self.grad_blend * 0 if self.train_blending else None

    #################################################################

    def return_grads(self) -> np.ndarray:
        """
        Returns the calculated gradients for trainable parameters.

        Returns:
        --------
        np.ndarray:
            A concatenated array of all gradients for trainable parameters.
        """
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
        if grads is not None:
            if self.grad_cen_up is not None:
                grads = np.concatenate((grads, self.grad_cen_up.reshape((-1, 1))))
                grads = np.concatenate((grads, self.grad_cen_low.reshape((-1, 1))))
            if self.grad_var_up is not None:
                grads = np.concatenate((grads, self.grad_var_up.reshape((-1, 1))))
                grads = np.concatenate((grads, self.grad_var_low.reshape((-1, 1))))
            if self.grad_blend is not None:
                grads = np.concatenate((grads, self.grad_blend.reshape((-1, 1))))
        return grads

    #################################################################

    def backward(self, batch_index: int, seq_index: int, error: np.ndarray) -> np.ndarray:
        """
        Computes gradients and propagates errors back to the previous layer.

        Parameters:
        -----------
        batch_index : int
            Index of the current batch.
        seq_index : int
            Index of the current sequence in the batch.
        error : np.ndarray
            Error signal from the subsequent layer or time step.

        Returns:
        --------
        np.ndarray:
            Propagated error for the input, reshaped to match input dimensions.
        """
        # Update gradient for blending factor
        if self.train_blending:
            self.grad_blend += error.reshape((-1, 1)) * (self.upper_output[batch_index, seq_index] - 
                                                        self.lower_output[batch_index, seq_index])

        # Scale errors using the blending factor
        e_max = self.blending_factor * error.reshape((-1, 1))
        e_min = (1 - self.blending_factor) * error.reshape((-1, 1))

        # Distribute error to upper and lower layers
        e_upper = e_max * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))) + \
                e_min * self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))
        e_lower = e_min * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))) + \
                e_max * self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))

        # Compute gradients for centers
        if self.train_center:
            self.grad_cen_up += np.diag(e_upper.ravel() * self.upper_output[batch_index, seq_index].ravel() * 
                                        self.upper_var.ravel() ** -2) @ \
                                (np.repeat(self.input[batch_index, seq_index].reshape(1, -1), self.output_size, axis=0) - 
                                self.upper_center)

            self.grad_cen_low += np.diag(e_lower.ravel() * self.lower_output[batch_index, seq_index].ravel() * 
                                        self.lower_var.ravel() ** -2) @ \
                                (np.repeat(self.input[batch_index, seq_index].reshape(1, -1), self.output_size, axis=0) - 
                                self.lower_center)

        # Compute gradients for variances
        if self.train_var:
            self.grad_var_up += (e_upper.ravel() * self.upper_net[batch_index, seq_index].ravel() ** 2 * 
                                self.upper_var.ravel() ** -3 * 
                                self.upper_output[batch_index, seq_index].ravel()).reshape((-1, 1))
            self.grad_var_low += (e_lower.ravel() * self.lower_net[batch_index, seq_index].ravel() ** 2 * 
                                self.lower_var.ravel() ** -3 * 
                                self.lower_output[batch_index, seq_index].ravel()).reshape((-1, 1))

        # Compute propagated error
        error_x_up = (e_upper.ravel() * self.upper_output[batch_index, seq_index].ravel() * 
                    self.upper_var.ravel() ** -2).reshape((-1, 1))
        e_x_up = (np.ones((self.input_size, self.output_size)) @ error_x_up).ravel() * \
                (np.sum(2 * self.upper_center, axis=0).ravel() - 4 * self.input[batch_index, seq_index].ravel())

        error_x_low = (e_lower.ravel() * self.lower_output[batch_index, seq_index].ravel() * 
                    self.lower_var.ravel() ** -2).reshape((-1, 1))
        e_x_low = (np.ones((self.input_size, self.output_size)) @ error_x_low).ravel() * \
                (np.sum(2 * self.lower_center, axis=0).ravel() - 4 * self.input[batch_index, seq_index].ravel())

        error_in = e_x_low + e_x_up
        return error_in.reshape((-1, 1))
