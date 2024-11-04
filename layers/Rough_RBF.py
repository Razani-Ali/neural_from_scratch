import numpy as np
from initializers.weight_initializer import RBF_weight_init
from optimizers.set_optimizer import init_optimizer


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
                 var_init_method: str = 'average', var_init_const: float = 1.0,
                 train_blending: bool = True, center_init_method: str = 'random',
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
        
        # Initialize upper and lower network centers
        self.upper_center = RBF_weight_init(input_size, output_size, method=center_init_method,
                                            distribution=center_distribution, ranges=center_uniform_range,
                                            var=center_normal_var, data=data)
        self.lower_center = RBF_weight_init(input_size, output_size, method=center_init_method,
                                            distribution=center_distribution, ranges=center_uniform_range,
                                            var=center_normal_var, data=data)

        # Initialize alpha (blend factor) with a default value of 0.5
        self.blending_factor = np.full((output_size, 1), 0.5)

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
            self.final_output[batch_index] = self.blending_factor * self.upper_output[batch_index] + \
                                             (1 - self.blending_factor) * self.lower_output[batch_index]
        print((self.final_output[:input.shape[0]].reshape((-1, self.output_size))).shape)
        return self.final_output[:input.shape[0]].reshape((-1, self.output_size))

    #################################################################

    def optimzer_init(self, optimizer='Adam', **kwargs) -> None:
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, grads: np.ndarray, learning_rate: float = 1e-3) -> None:
        deltas = self.Optimizer(grads, learning_rate)
        ind2 = 0
        if self.train_center:
            ind1 = ind2
            ind2 += int(np.size(self.upper_center))
            delta_cen = deltas[ind1:ind2].reshape(self.upper_center.shape)
            self.upper_center -= delta_cen
            ind1 = ind2
            ind2 += int(np.size(self.lower_center))
            delta_cen = deltas[ind1:ind2].reshape(self.lower_center.shape)
            self.lower_center -= delta_cen
        if self.train_var:
            ind1 = ind2
            ind2 += np.size(self.upper_var)
            delta_var = deltas[ind1:ind2].reshape(self.upper_var.shape)
            self.upper_var -= delta_var
            ind1 = ind2
            ind2 += np.size(self.lower_var)
            delta_var = deltas[ind1:ind2].reshape(self.lower_var.shape)
            self.lower_var -= delta_var
        if self.train_blending:
            ind1 = ind2
            ind2 += np.size(self.blending_factor)
            delta_blend = deltas[ind1:ind2].reshape(self.blending_factor.shape)
            self.blending_factor -= delta_blend

    #################################################################

    def backward(self, error_batch: np.ndarray, learning_rate: float = 1e-3, return_error=False, return_grads=False, modify=True):
        if return_error:
            error_in = np.zeros(self.input.shape)  # Initialize error output array with shape of input

        grad_c_up = np.zeros(self.upper_center.shape) if self.train_center else None
        grad_c_low = np.zeros(self.lower_center.shape) if self.train_center else None
        grad_var_up = np.zeros(self.upper_var.shape) if self.train_var else None
        grad_var_low = np.zeros(self.lower_var.shape) if self.train_var else None
        grad_alpha = np.zeros(self.blending_factor.shape) if self.train_blending else None

        # Iterate over each batch index to compute gradients based on batch errors
        for batch_index, one_batch_error in enumerate(error_batch):
            if self.train_blending:
                # Compute gradient for alpha using the difference between upper and lower outputs
                grad_alpha += one_batch_error.reshape((-1, 1)) * \
                    (self.upper_output[batch_index] - self.lower_output[batch_index])

            e_max = self.blending_factor * one_batch_error.reshape((-1, 1))  # Compute maximum error component
            e_min = (1 - self.blending_factor) * one_batch_error.reshape((-1, 1))  # Compute minimum error component

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
            if return_error:
                error_x_up = (e_upper.ravel() * self.upper_output[batch_index].ravel() * \
                           self.upper_var.ravel() ** -2).reshape((-1, 1))
                e_x_up = (np.ones((self.input_size, self.output_size)) @ error_x_up).ravel() *\
                    (np.sum(2 * self.upper_center, axis=0).ravel() - 4 * self.input[batch_index].ravel())
            
                # Calculate error contribution for lower centers based on input and center deviation
                error_x_low = (e_lower.ravel() * self.lower_output[batch_index].ravel() * \
                        self.lower_var.ravel() ** -2).reshape((-1, 1))
                e_x_low = (np.ones((self.input_size, self.output_size)) @ error_x_low).ravel() *\
                    (np.sum(2 * self.lower_center, axis=0).ravel() - 4 * self.input[batch_index].ravel())
                
                error_in[batch_index] = e_x_low + e_x_up  # Summing error contributions for each batch

        if self.train_center:
            # Normalize center gradients by batch size if centers are trainable
            grad_c_up /= error_batch.shape[0]
            grad_c_low /= error_batch.shape[0]

        if self.train_var:
            # Normalize variance gradients by batch size if variances are trainable
            grad_var_up /= error_batch.shape[0]
            grad_var_low /= error_batch.shape[0]

        if self.train_blending:
            # Normalize alpha gradient by batch size if alpha is trainable
            grad_alpha /= error_batch.shape[0]

        # Update the parameters using the computed gradients
        grads = None if (grad_c_up is None) and (grad_var_up is None) else np.array([]).reshape((-1,1))
        if grads is not None:
            if grad_c_up is not None:
                grads = np.concatenate((grads, grad_c_up.reshape((-1,1))))
                grads = np.concatenate((grads, grad_c_low.reshape((-1,1))))
            if grad_var_up is not None:
                grads = np.concatenate((grads, grad_var_up.reshape((-1,1))))
                grads = np.concatenate((grads, grad_var_low.reshape((-1,1))))
            if grad_alpha is not None:
                grads = np.concatenate((grads, grad_alpha.reshape((-1,1))))
        if modify:
            self.update(grads, learning_rate=learning_rate)

        # Return parameters gradients and gradient with respect to input if needed
        if return_error and return_grads:
            return {'error_in': error_in, 'gradients': grads}
        elif return_error:
            return error_in
        elif return_grads:
            return grads
