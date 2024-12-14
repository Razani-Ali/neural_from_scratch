import numpy as np
from initializers.weight_initializer import RBF_weight_init
from optimizers.set_optimizer import init_optimizer


class RBFRoughKernel:
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
    optimizer_init():
        Initializes optimizer parameters for current layer.
    update(...):
        Updates the network parameters using gradients and Adam or SGD optimizer.
    backward(...):
        Backpropagates the error and computes gradients for network parameters.
    """

    def __init__(self, input_size: int, output_size: int, batch_size: int = 32,
                 train_center: bool = True, train_var: bool = True, data=None,
                 var_init_method: str = 'max', var_init_const: float = 1.0,
                 train_blending: bool = True, center_init_method: str = 'random',
                 center_distribution: str = 'uniform',
                 center_uniform_range: tuple = (-1, 1), center_normal_var: float = 1.0):

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
        """
        
        # Store configuration and initialize centers with helper function
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.train_center = train_center
        self.train_var = train_var
        self.train_blending = train_blending
        self.activation = 'Guassian Kernel'
        
        # Initialize upper and lower network centers
        self.upper_center = RBF_weight_init(input_size, output_size, method=center_init_method,
                                            distribution=center_distribution, ranges=center_uniform_range,
                                            var=center_normal_var, data=data)
        self.lower_center = RBF_weight_init(input_size, output_size, method=center_init_method,
                                            distribution=center_distribution, ranges=center_uniform_range,
                                            var=center_normal_var, data=data)
        # self.lower_center = self.upper_center + np.random.normal(scale=0.1, size=self.upper_center.shape)

        # Initialize alpha (blend factor) with a default value of 0.5
        self.blending_factor = np.full((output_size, ), 0.5)

        # Initialize variances based on the chosen method
        if var_init_method == 'constant':
            self.var = np.full((output_size, ), var_init_const)
        elif var_init_method == 'average':
            self.var = np.mean(self.upper_center, axis=1) / output_size
        elif var_init_method == 'max':
            self.var = np.max(self.upper_center, axis=1) / np.sqrt(2 * output_size)
        else:
            raise ValueError('your variance initialization is not supported')

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
        Calculates the total number of trainable parameters in the network.

        The trainable parameters include:
        - Upper and lower centers if `train_center` is True.
        - Variances if `train_var` is True.
        - Blending factors (alpha) if `train_blending` is True.

        Returns:
        --------
        int
            The total count of trainable parameters.
        """
        # Initialize parameter counter
        params = 0

        # Add the size of the centers (upper and lower) if trainable
        if self.train_center:
            params += 2 * np.size(self.upper_center)  # Upper and lower centers

        # Add the size of the variances if trainable
        if self.train_var:
            params += np.size(self.var)  # Variances

        # Add the size of the blending factors if trainable
        if self.train_blending:
            params += np.size(self.blending_factor)  # Alpha (blending factors)

        # Return the total count of trainable parameters
        return params
    
    #################################################################

    def all_params(self) -> int:
        """
        Calculates the total number of parameters in the network.

        This includes:
        - Both upper and lower centers.
        - Variances.
        - Blending factors (alpha).

        Returns:
        --------
        int
            The total count of parameters in the network.
        """
        # Total parameters include both upper and lower centers,
        # variances, and blending factors
        return 2 * np.size(self.upper_center) + np.size(self.var) + np.size(self.blending_factor)

    #################################################################

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the RBF network.

        This computes the distances between the input data and the centers of the 
        upper and lower networks, calculates the Gaussian kernel outputs, and 
        combines them using the blending factor (alpha).

        Parameters:
        -----------
        input : np.ndarray
            Input data array with shape (batch_size, input_size).

        Returns:
        --------
        np.ndarray
            The output of the RBF network with shape (batch_size, output_size).
        """
        self.input = input  # Store input for this forward pass

        # Ensure the batch size does not exceed the model's configuration
        if self.batch_size < input.shape[0]:
            raise ValueError('Data batch size cannot be larger than model batch size')

        # Iterate through each batch sample
        for batch_index, input_vector in enumerate(input):
            input_vector = input_vector.reshape(1, -1)
            # Compute distances between the input vector and upper/lower centers
            self.upper_net[batch_index] = np.linalg.norm(
                (np.repeat(input_vector, self.output_size, axis=0) - self.upper_center),
                axis=1
            ).ravel()
            self.lower_net[batch_index] = np.linalg.norm(
                (np.repeat(input_vector, self.output_size, axis=0) - self.lower_center),
                axis=1
            ).ravel()

            # Compute Gaussian kernel outputs for upper and lower networks
            try:
                self.upper_output[batch_index] = np.exp(
                    -0.5 * np.square(self.upper_net[batch_index] / self.var)
                )
                self.lower_output[batch_index] = np.exp(
                    -0.5 * np.square(self.lower_net[batch_index] / self.var)
                )
            except FloatingPointError:
                # Handle potential division by zero with a small constant
                self.upper_output[batch_index] = np.exp(
                    -0.5 * np.square(self.upper_net[batch_index] / (self.var + 1e-7))
                )
                self.lower_output[batch_index] = np.exp(
                    -0.5 * np.square(self.lower_net[batch_index] / (self.var + 1e-7))
                )

            # Combine upper and lower outputs using the blending factor
            up_out = self.upper_output[batch_index].reshape((-1,1))
            low_out = self.lower_output[batch_index].reshape((-1,1))
            concat_out = np.concatenate((up_out, low_out), axis=1)  # Concatenate outputs

            # Determine the maximum and minimum outputs across upper and lower networks
            self.minmax_reverse_stat[batch_index] = np.argmax(concat_out, axis=1)  # Track max source
            self.upper_output[batch_index] = np.max(concat_out, axis=1)  # Max output
            self.lower_output[batch_index] = np.min(concat_out, axis=1)  # Min output

            # Compute the final blended output using the alpha parameter
            self.final_output[batch_index] = (
                self.blending_factor * self.upper_output[batch_index] +
                (1 - self.blending_factor) * self.lower_output[batch_index]
            )

        # Return the final output for all input samples
        batch_index += 1
        return self.final_output[:batch_index]

    #################################################################
    
    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for the RBF network.

        This method sets up the optimizer to update trainable parameters during 
        training, such as centers, variances, and blending factors.

        Parameters:
        -----------
        optimizer : str, optional
            Name of the optimizer to use (default is 'Adam'). Examples include:
            - 'Adam': Adaptive Moment Estimation
            - 'SGD': Stochastic Gradient Descent
        **kwargs : dict, optional
            Additional configuration parameters for the optimizer (e.g., learning rate).

        Returns:
        --------
        None
        """
        # Call an external function to initialize the optimizer
        # Pass the total number of trainable parameters and the optimizer method
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, grads: np.ndarray, learning_rate: float = 1e-3) -> None:
        """
        Update model parameters using computed gradients and specified learning rate.

        The method uses the optimizer to calculate parameter deltas from gradients,
        and then applies these deltas to update the trainable parameters (centers,
        variances, and blending factor).

        Parameters:
        -----------
        grads : np.ndarray
            Gradients for model parameters, flattened into a single array.
        learning_rate : float, optional
            Learning rate for parameter updates (default is 0.001).

        Returns:
        --------
        None
        """
        # Step 1: Compute delta updates using the optimizer
        # The optimizer function uses the gradients and learning rate to calculate deltas
        deltas = self.Optimizer(grads, learning_rate)

        # Step 2: Initialize index pointer for slicing the deltas array
        ind2 = 0

        # Step 3: Update centers if training is enabled
        if self.train_center:
            # Upper centers:
            # Slice the appropriate section of deltas, reshape it, and update the upper centers
            ind1 = ind2  # Start index for upper centers
            ind2 += int(np.size(self.upper_center))  # End index for upper centers
            delta_cen = deltas[ind1:ind2].reshape(self.upper_center.shape)  # Reshape delta for upper centers
            self.upper_center -= delta_cen  # Update upper centers

            # Lower centers:
            # Slice the next section of deltas, reshape it, and update the lower centers
            ind1 = ind2  # Start index for lower centers
            ind2 += int(np.size(self.lower_center))  # End index for lower centers
            delta_cen = deltas[ind1:ind2].reshape(self.lower_center.shape)  # Reshape delta for lower centers
            self.lower_center -= delta_cen  # Update lower centers

        # Step 4: Update variances if training is enabled
        if self.train_var:
            # Slice the next section of deltas, reshape it, and update the variances
            ind1 = ind2  # Start index for variances
            ind2 += np.size(self.var)  # End index for variances
            delta_var = deltas[ind1:ind2].reshape(self.var.shape)  # Reshape delta for variances
            self.var -= delta_var  # Update variances

        # Step 5: Update blending factor (alpha) if training is enabled
        if self.train_blending:
            # Slice the next section of deltas, reshape it, and update the blending factor
            ind1 = ind2  # Start index for blending factor
            ind2 += np.size(self.blending_factor)  # End index for blending factor
            delta_blend = deltas[ind1:ind2].reshape(self.blending_factor.shape)  # Reshape delta for blending factor
            self.blending_factor -= delta_blend  # Update blending factor

    #################################################################

    def backward(
        self,
        error_batch: np.ndarray,
        learning_rate: float = 1e-3,
        return_error: bool = False,
        return_grads: bool = False,
        modify: bool = True
    ):
        """
        Perform backward propagation to calculate gradients and update parameters.

        This method calculates gradients for trainable parameters (centers, variances,
        and blending factor), updates parameters if `modify` is True, and optionally
        propagates error to the previous layer.

        Parameters:
        -----------
        error_batch : np.ndarray
            Batch of errors for each output node, shape (batch_size, output_size).
        learning_rate : float, optional
            Learning rate for updating parameters (default is 0.001).
        return_error : bool, optional
            If True, returns error propagated to the previous layer (default is False).
        return_grads : bool, optional
            If True, returns computed gradients (default is False).
        modify : bool, optional
            If True, updates parameters using gradients (default is True).

        Returns:
        --------
        Optional[Union[Dict[str, Any], np.ndarray]]:
            If specified, returns a dictionary with `error_in` (error to the previous
            layer) and/or `gradients`. Otherwise, returns None.
        """
        # Step 1: Initialize propagated error and gradients
        if return_error:
            error_in = np.zeros(self.input.shape)  # Error to propagate to previous layer
        grad_c_up = np.zeros(self.upper_center.shape) if self.train_center else None  # Gradient for upper centers
        grad_c_low = np.zeros(self.lower_center.shape) if self.train_center else None  # Gradient for lower centers
        grad_var = np.zeros(self.var.shape) if self.train_var else None  # Gradient for variances
        grad_alpha = np.zeros(self.blending_factor.shape) if self.train_blending else None  # Gradient for blending factor

        # Step 2: Loop through the batch to calculate gradients
        for batch_index, one_batch_error in enumerate(error_batch):
            one_batch_error =one_batch_error

            # 2.1: Compute gradient for blending factor (alpha)
            if self.train_blending:
                grad_alpha += one_batch_error * (
                    self.upper_output[batch_index] - self.lower_output[batch_index]
                )

            # 2.2: Distribute error to upper and lower layers
            e_max = self.blending_factor * one_batch_error
            e_min = (1 - self.blending_factor) * one_batch_error
            e_upper = (
                e_max * np.logical_not(self.minmax_reverse_stat[batch_index]) +
                e_min * self.minmax_reverse_stat[batch_index]
            )
            e_lower = (
                e_min * np.logical_not(self.minmax_reverse_stat[batch_index]) +
                e_max * self.minmax_reverse_stat[batch_index]
            )

            # 2.3: Compute gradients for centers
            if self.train_center:
                grad_c_up += np.diag(
                    e_upper * self.upper_output[batch_index] * self.var ** -2
                ) @ (np.repeat(self.input[batch_index].reshape(1, -1), self.output_size, axis=0) - self.upper_center)
                grad_c_low += np.diag(
                    e_lower * self.lower_output[batch_index] * self.var ** -2
                ) @ (np.repeat(self.input[batch_index].reshape(1, -1), self.output_size, axis=0) - self.lower_center)

            # 2.4: Compute gradients for variances
            if self.train_var:
                grad_var += (
                    e_upper.ravel() * self.upper_net[batch_index] ** 2 * self.var ** -3 *
                    self.upper_output[batch_index]
                )
                grad_var += (
                    e_lower.ravel() * self.lower_net[batch_index] ** 2 * self.var ** -3 *
                    self.lower_output[batch_index]
                )

            # 2.5: Compute propagated error if required
            if return_error:
                error_x_up = (
                    e_upper.ravel() * self.upper_output[batch_index] * self.var ** -2
                ).reshape((-1, 1))
                e_x_up = (np.ones((self.input_size, self.output_size)) @ error_x_up).ravel() * (
                    np.sum(2 * self.upper_center, axis=0).ravel() - 4 * self.input[batch_index]
                )

                error_x_low = (
                    e_lower.ravel() * self.lower_output[batch_index] * self.var ** -2
                ).reshape((-1, 1))
                e_x_low = (np.ones((self.input_size, self.output_size)) @ error_x_low).ravel() * (
                    np.sum(2 * self.lower_center, axis=0).ravel() - 4 * self.input[batch_index]
                )
                error_in[batch_index] = e_x_low + e_x_up  # Combine errors from upper and lower networks

        # Step 3: Normalize gradients by batch size
        if self.train_center:
            grad_c_up /= error_batch.shape[0]
            grad_c_low /= error_batch.shape[0]
        if self.train_var:
            grad_var /= error_batch.shape[0]
        if self.train_blending:
            grad_alpha /= error_batch.shape[0]

        # Step 4: Concatenate gradients and update parameters if `modify` is True
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
        if grads is not None:
            if grad_c_up is not None:
                grads = np.concatenate((grads, grad_c_up.reshape((-1, 1))))
                grads = np.concatenate((grads, grad_c_low.reshape((-1, 1))))
            if grad_var is not None:
                grads = np.concatenate((grads, grad_var.reshape((-1, 1))))
            if grad_alpha is not None:
                grads = np.concatenate((grads, grad_alpha.reshape((-1, 1))))
        if modify:
            self.update(grads, learning_rate=learning_rate)  # Update parameters using gradients

        # Step 5: Return required outputs
        if return_error and return_grads:
            return {'error_in': error_in, 'gradients': grads}  # Return both propagated error and gradients
        elif return_error:
            return error_in  # Return only propagated error
        elif return_grads:
            return grads  # Return only gradients

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class TimeRBFRoughKernel:
    """
    A rough Radial Basis Function (RBF) network with dual-layer structure 
    (upper and lower networks) and an optional adaptive blending factor (alpha).

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
    blending_factor : np.ndarray
        Alpha parameter blending the outputs of upper and lower networks.
    var : np.ndarray
        Variances of the RBF neurons.

    Methods:
    --------
    trainable_params() -> int:
        Returns the total number of trainable parameters.
    __call__(batch_index: int, seq_index: int, input: np.ndarray) -> np.ndarray:
        Forward pass through the network.
    optimizer_init(optimizer: str = 'Adam', **kwargs) -> None:
        Initializes optimizer parameters for the current layer.
    update(batch_size: int, learning_rate: float, grads: Optional[np.ndarray]) -> None:
        Updates the network parameters using gradients.
    return_grads() -> Optional[np.ndarray]:
        Returns the computed gradients for trainable parameters.
    backward(batch_index: int, seq_index: int, error: np.ndarray) -> np.ndarray:
        Backpropagates the error and computes gradients for network parameters.
    """

    def __init__(
        self,
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
        center_distribution: str = 'uniform',
        center_uniform_range: tuple = (-1, 1),
        center_normal_var: float = 1.0
    ):
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
        """
        # Input and output configurations
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.train_center = train_center
        self.train_var = train_var
        self.train_blending = train_blending
        self.activation = 'Gaussian Kernel'

        # Initialize centers for upper and lower networks
        self.upper_center = RBF_weight_init(
            input_size, output_size, method=center_init_method,
            distribution=center_distribution, ranges=center_uniform_range,
            var=center_normal_var, data=data
        )
        self.lower_center = RBF_weight_init(
            input_size, output_size, method=center_init_method,
            distribution=center_distribution, ranges=center_uniform_range,
            var=center_normal_var, data=data
        )

        # Initialize blending factor (alpha)
        self.blending_factor = np.full((output_size, 1), 0.5)

        # Initialize variances
        if var_init_method == 'constant':
            self.var = np.full((output_size, 1), var_init_const)
        elif var_init_method == 'average':
            self.var = np.mean(self.upper_center, axis=1).reshape((-1, 1)) / output_size
        elif var_init_method == 'max':
            self.var = np.max(self.upper_center, axis=1).reshape((-1, 1)) / np.sqrt(2 * output_size)
        else:
            raise ValueError('Invalid variance initialization method.')

        # Prepare storage for network outputs and intermediate states
        self.upper_net = np.zeros((batch_size, time_steps, output_size, 1))
        self.lower_net = np.zeros((batch_size, time_steps, output_size, 1))
        self.minmax_reverse_stat = np.zeros((batch_size, time_steps, output_size))
        self.output = np.zeros((batch_size, time_steps, output_size, 1))
        self.input = np.zeros((batch_size, time_steps, input_size, 1))
        self.upper_output = np.zeros((batch_size, time_steps, output_size, 1))
        self.lower_output = np.zeros((batch_size, time_steps, output_size, 1))

        # Gradients for parameters
        self.grad_cen_up = np.zeros(self.upper_center.shape) if self.train_center else None
        self.grad_cen_low = np.zeros(self.lower_center.shape) if self.train_center else None
        self.grad_var = np.zeros(self.var.shape) if self.train_var else None
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
            params += np.size(self.var)  # Variances
        if self.train_blending:
            params += np.size(self.blending_factor)  # Blending factor (alpha)
        return params

    #################################################################

    def all_params(self) -> int:
        """
        Calculates the total number of parameters in the network.

        Returns:
        --------
        int
            The total count of parameters in the network.
        """
        return 2 * np.size(self.upper_center) + np.size(self.var) + np.size(self.blending_factor)

    #################################################################

    def __call__(self, batch_index: int, seq_index: int, input: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the layer.

        Parameters:
        -----------
        batch_index : int
            Current batch index.
        seq_index : int
            Current sequence index.
        input : np.ndarray
            Input data array of shape (batch_size, time_steps, input_size, 1).

        Returns:
        --------
        np.ndarray
            Output data array of shape (batch_size, output_size, 1).
        """
        self.input[batch_index, seq_index] = input.reshape((-1, 1))

        # Calculate distances between input vector and centers for both networks
        self.upper_net[batch_index, seq_index] = np.linalg.norm(
            (np.repeat(input.reshape(1, -1), self.output_size, axis=0) - self.upper_center), axis=1
        ).reshape((-1, 1))
        self.lower_net[batch_index, seq_index] = np.linalg.norm(
            (np.repeat(input.reshape(1, -1), self.output_size, axis=0) - self.lower_center), axis=1
        ).reshape((-1, 1))

        # Compute RBF output for upper and lower networks with numerical stability handling
        try:
            self.upper_output[batch_index, seq_index] = np.exp(-0.5 * np.square(
                self.upper_net[batch_index, seq_index].ravel() / self.var.ravel()
            )).reshape(-1, 1)
            self.lower_output[batch_index, seq_index] = np.exp(-0.5 * np.square(
                self.lower_net[batch_index, seq_index].ravel() / self.var.ravel()
            )).reshape(-1, 1)
        except FloatingPointError:
            self.upper_output[batch_index, seq_index] = np.exp(-0.5 * np.square(
                self.upper_net[batch_index, seq_index].ravel() / (self.var.ravel() + 1e-7)
            )).reshape(-1, 1)
            self.lower_output[batch_index, seq_index] = np.exp(-0.5 * np.square(
                self.lower_net[batch_index, seq_index].ravel() / (self.var.ravel() + 1e-7)
            )).reshape(-1, 1)

        # Combine upper and lower outputs and compute the final output using blending factor
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
        Initializes the optimizer for the Rough RBF model using the specified method.

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
        Updates the layer's weights and biases based on calculated gradients.

        Parameters:
        -----------
        batch_size : int
            Batch size used to normalize gradients.
        learning_rate : float, optional
            Learning rate for parameter updates (default is 1e-3).
        grads : np.ndarray, optional
            Precomputed gradients for parameters (default is None).

        Returns:
        --------
        None
        """
        if grads is None:
            # If no gradients provided, prepare gradients from stored values
            grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
            if grads is not None:
                if self.grad_cen_up is not None:
                    grads = np.concatenate((grads, self.grad_cen_up.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_cen_low.reshape((-1, 1))))
                if self.grad_var is not None:
                    grads = np.concatenate((grads, self.grad_var.reshape((-1, 1))))
                if self.grad_blend is not None:
                    grads = np.concatenate((grads, self.grad_blend.reshape((-1, 1))))
                grads /= batch_size  # Normalize gradients by batch size

        # Apply the optimizer to compute parameter updates (deltas)
        deltas = self.Optimizer(grads, learning_rate)
        ind2 = 0  # Initialize index pointer for deltas

        # Update upper and lower centers if trainable
        if self.train_center:
            ind1 = ind2
            ind2 += int(np.size(self.upper_center))
            delta_cen_up = deltas[ind1:ind2].reshape(self.upper_center.shape)
            self.upper_center -= delta_cen_up  # Update upper centers

            ind1 = ind2
            ind2 += int(np.size(self.lower_center))
            delta_cen_low = deltas[ind1:ind2].reshape(self.lower_center.shape)
            self.lower_center -= delta_cen_low  # Update lower centers

        # Update variances if trainable
        if self.train_var:
            ind1 = ind2
            ind2 += int(np.size(self.var))
            delta_var = deltas[ind1:ind2].reshape(self.var.shape)
            self.var -= delta_var  # Update variances

        # Update blending factor if trainable
        if self.train_blending:
            ind1 = ind2
            ind2 += int(np.size(self.blending_factor))
            delta_blend = deltas[ind1:ind2].reshape(self.blending_factor.shape)
            self.blending_factor -= delta_blend  # Update blending factor

        # Reset stored gradients after applying updates
        self.grad_cen_up = self.grad_cen_up * 0 if self.train_center else None
        self.grad_cen_low = self.grad_cen_low * 0 if self.train_center else None
        self.grad_var = self.grad_var * 0 if self.train_var else None
        self.grad_blend = self.grad_blend * 0 if self.train_blending else None

    #################################################################

    def return_grads(self) -> np.ndarray:
        """
        Returns the computed gradients for all trainable parameters.

        Returns:
        --------
        np.ndarray
            Concatenated gradients for all trainable parameters.
        """
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
        if grads is not None:
            if self.grad_cen_up is not None:
                grads = np.concatenate((grads, self.grad_cen_up.reshape((-1, 1))))
                grads = np.concatenate((grads, self.grad_cen_low.reshape((-1, 1))))
            if self.grad_var is not None:
                grads = np.concatenate((grads, self.grad_var.reshape((-1, 1))))
            if self.grad_blend is not None:
                grads = np.concatenate((grads, self.grad_blend.reshape((-1, 1))))
        return grads

    #################################################################

    def backward(self, batch_index: int, seq_index: int, error: np.ndarray) -> np.ndarray:
        """
        Computes gradients for parameters and optionally propagates error to the previous layer.

        Parameters:
        -----------
        batch_index : int
            Current batch index.
        seq_index : int
            Current sequence index.
        error : np.ndarray
            Error signal from the subsequent layer.

        Returns:
        --------
        np.ndarray
            Propagated error to the previous layer.
        """
        # Compute gradient for blending factor
        if self.train_blending:
            self.grad_blend += error.reshape((-1, 1)) * (
                self.upper_output[batch_index, seq_index] - self.lower_output[batch_index, seq_index]
            )

        # Scale error based on blending factor
        e_max = self.blending_factor * error.reshape((-1, 1))
        e_min = (1 - self.blending_factor) * error.reshape((-1, 1))

        # Split error into upper and lower contributions
        e_upper = e_max * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))) + \
                e_min * self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))
        e_lower = e_min * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))) + \
                e_max * self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))

        # Compute gradients for centers
        if self.train_center:
            self.grad_cen_up += np.diag(e_upper.ravel() * self.upper_output[batch_index, seq_index].ravel() * self.var.ravel() ** -2) @ \
                                (np.repeat(self.input[batch_index, seq_index].reshape(1, -1), self.output_size, axis=0) - self.upper_center)
            self.grad_cen_low += np.diag(e_lower.ravel() * self.lower_output[batch_index, seq_index].ravel() * self.var.ravel() ** -2) @ \
                                (np.repeat(self.input[batch_index, seq_index].reshape(1, -1), self.output_size, axis=0) - self.lower_center)

        # Compute gradients for variances
        if self.train_var:
            self.grad_var += (e_upper.ravel() * self.upper_net[batch_index, seq_index].ravel() ** 2 * self.var.ravel() ** -3 *
                            self.upper_output[batch_index, seq_index].ravel()).reshape((-1, 1))
            self.grad_var += (e_lower.ravel() * self.lower_net[batch_index, seq_index].ravel() ** 2 * self.var.ravel() ** -3 *
                            self.lower_output[batch_index, seq_index].ravel()).reshape((-1, 1))

        # Compute propagated error to the input layer
        error_x_up = (e_upper.ravel() * self.upper_output[batch_index, seq_index].ravel() * self.var.ravel() ** -2).reshape((-1, 1))
        error_x_low = (e_lower.ravel() * self.lower_output[batch_index, seq_index].ravel() * self.var.ravel() ** -2).reshape((-1, 1))

        e_x_up = (np.ones((self.input_size, self.output_size)) @ error_x_up).ravel() * (
            np.sum(2 * self.upper_center, axis=0).ravel() - 4 * self.input[batch_index, seq_index].ravel()
        )
        e_x_low = (np.ones((self.input_size, self.output_size)) @ error_x_low).ravel() * (
            np.sum(2 * self.lower_center, axis=0).ravel() - 4 * self.input[batch_index, seq_index].ravel()
        )

        # Combine contributions from upper and lower networks
        error_in = e_x_low + e_x_up
        return error_in.reshape((-1, 1))
