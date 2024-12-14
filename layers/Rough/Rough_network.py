import numpy as np
from activations.activation_functions import net2out, net2Fprime
from initializers.weight_initializer import Dense_weight_init
from optimizers.set_optimizer import init_optimizer


class Rough:
    """
    Rough class implements a neural network layer-like structure with an additional alpha parameter to control
    the weighting between upper and lower networks, initialized with separate weight matrices and biases.
    
    Parameters:
    ----------
    input_size : int
        The number of input neurons or features.
    output_size : int
        The number of output neurons.
    use_bias : bool, optional
        Whether to use bias in the upper and lower networks (default is True).
    batch_size : int, optional
        The number of samples per batch (default is 32).
    train_bias: bool, optional
        Whether to train bias if use_bias or not
    train_weights: bool, optional
        Whether to train weights or not
    train_blending : bool, optional
        If True, blending factor or weight of average will be trained (default is False).
    activation : str, optional
        Activation function to use (default is 'sigmoid'). Other options include 'relu', 'tanh', etc.
    alpha_acti : float, optional
        A scaling parameter for the activation function (default is None).
    weights_uniform_range : tuple, optional
        The range for initializing weights uniformly (default is (-1, 1)).
    L2_coe : float, optional
        L2 regularization coefficient
    L1_coe : float, optional
        L1 regularization coefficient
    
    """

    def __init__(self, input_size: int, output_size: int, use_bias: bool = True, batch_size: int = 32,
                 train_weights: bool = True, train_bias: bool = True, train_blending: bool = False,
                 activation: str = 'sigmoid', alpha_acti: float = None, weights_uniform_range: tuple = (-1, 1),
                 L2_coe: float = 0.0, L1_coe: float = 0.0):

        self.output_size = output_size  # Number of output neurons
        self.input_size = input_size  # Number of input neurons/features
        self.batch_size = batch_size  # Batch size
        self.L2_coe = L2_coe  # L2 regularization coefficient
        self.L1_coe = L1_coe  # L1 regularization coefficient
        self.use_bias = use_bias  # Whether to use bias
        self.activation = activation  # Activation function to use
        self.alpha_activation = alpha_acti  # Alpha activation scaling parameter
        self.train_weights = train_weights
        self.train_bias = False if use_bias is False else train_bias
        self.train_blending = train_blending


        # Split weight initialization ranges into upper and lower halves
        middle = (weights_uniform_range[0] + weights_uniform_range[1]) / 2
        upper_range = (weights_uniform_range[0], middle)
        lower_range = (middle, weights_uniform_range[1])
        
        # Initialize upper and lower weights using uniform distribution
        self.upper_weight = Dense_weight_init(input_size, output_size, method="uniform", ranges=upper_range)
        self.lower_weight = Dense_weight_init(input_size, output_size, method="uniform", ranges=lower_range)
        
        # Initialize bias if specified
        if self.use_bias:
            self.upper_bias = np.zeros((output_size, 1))
            self.lower_bias = np.zeros((output_size, 1))

        # Initialize alpha (blend factor between upper and lower networks)
        self.blending_factor = np.zeros((output_size, 1)) + 0.5

        # Initialize network outputs for upper and lower nets
        self.upper_net = np.zeros((batch_size, output_size, 1))
        self.lower_net = np.zeros((batch_size, output_size, 1))

        # Used to store min-max reverse operation results
        self.minmax_reverse_stat = np.zeros((batch_size, output_size, 1))

        # Final and intermediate outputs
        self.final_output = np.zeros((batch_size, output_size, 1))
        self.upper_output = np.zeros((batch_size, output_size, 1))
        self.lower_output = np.zeros((batch_size, output_size, 1))

    #################################################################

    def trainable_params(self) -> int:
        """
        Returns the number of trainable parameters in the model.
        This includes both weights, biases (if enabled), and alpha (if trainable).
        """
        params = 0
        if self.train_weights:
            params += np.size(self.upper_weight) * 2  # We have both upper and lower weights
        if self.train_bias:
            params += np.size(self.upper_bias) * 2  # Add bias parameters if enabled
        if self.train_blending:
            params += np.size(self.blending_factor)  # Add alpha parameters if trainable

        return int(params)

    #################################################################

    def all_params(self) -> int:
        """
        Returns the number of parameters in the model.
        This includes both weights, biases (if enabled), and alpha (if trainable).
        """
        params = np.size(self.upper_weight) * 2 + np.size(self.blending_factor)
        if self.use_bias:
            params += np.size(self.upper_bias) * 2  # Add bias parameters if enabled

        return int(params)

    #################################################################

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass of the model.
        
        Parameters:
        -----------
        input : np.ndarray
            Input data, shape should be (batch_size, input_size).
        
        Returns:
        --------
        np.ndarray
            Output of the model, shape will be (batch_size, output_size).
        """
        self.input = input

        # Ensure batch size matches the input batch size
        if self.batch_size < input.shape[0]:
            raise ValueError('Data batch size cannot be larger than model batch size')

        # Loop through each sample in the batch
        for batch_index, input_vector in enumerate(input):
            input_vector = input_vector.reshape((-1, 1))
            # Compute net input for upper and lower networks
            self.upper_net[batch_index] = self.upper_weight @ input_vector
            self.lower_net[batch_index] = self.lower_weight @ input_vector

            # Add bias if applicable
            if self.use_bias:
                self.upper_net[batch_index] += self.upper_bias
                self.lower_net[batch_index] += self.lower_bias

            # Apply activation function to the net inputs
            up_out = net2out(self.upper_net[batch_index], self.activation, alpha=self.alpha_activation)
            low_out = net2out(self.lower_net[batch_index], self.activation, alpha=self.alpha_activation)

            # Concatenate upper and lower outputs to find min and max
            concat_out = np.concatenate((up_out, low_out), axis=1)
            self.minmax_reverse_stat[batch_index] = np.argmax(concat_out).reshape(-1,1)

            # Get max for upper and min for lower
            self.upper_output[batch_index] = np.max(concat_out, axis=1).reshape((-1, 1))
            self.lower_output[batch_index] = np.min(concat_out, axis=1).reshape((-1, 1))

            # Final output is a blend of upper and lower outputs, weighted by alpha
            self.final_output[batch_index] = self.blending_factor * self.upper_output[batch_index] +\
                (1 - self.blending_factor) * self.lower_output[batch_index]

        # Return the final output for all input samples
        batch_index += 1
        return self.final_output[:batch_index, :, 0]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for updating network parameters.

        Parameters:
        -----------
        optimizer : str, optional
            The type of optimizer to use, default is 'Adam'.
        **kwargs : dict, optional
            Additional parameters for configuring the optimizer.

        Returns:
        --------
        None
        """
        # Initialize the optimizer for this network layer, setting it to
        # update parameters based on the chosen method (e.g., Adam, SGD)
        # and any additional arguments provided (e.g., learning rate).
        # 'self.trainable_params()' gets all parameters that need updates.
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, grads: np.ndarray, learning_rate: float = 1e-3) -> None:
        """
        Applies calculated gradients to update trainable parameters.

        Parameters:
        -----------
        grads : np.ndarray
            Gradients for all trainable parameters.
        learning_rate : float, optional
            The rate at which parameters are adjusted, default is 1e-3.

        Returns:
        --------
        None
        """
        # Calculate updates for each parameter using the optimizer and provided gradients.
        deltas = self.Optimizer(grads, learning_rate)

        # Initialize an index to keep track of each parameter's position within deltas.
        ind2 = 0

        # Update weights if trainable
        if self.train_weights:
            # Update upper_weight
            ind1 = ind2
            ind2 += int(np.size(self.upper_weight))
            delta_w = deltas[ind1:ind2].reshape(self.upper_weight.shape)
            self.upper_weight -= delta_w  # Apply update

            # Update lower_weight
            ind1 = ind2
            ind2 += int(np.size(self.lower_weight))
            delta_w = deltas[ind1:ind2].reshape(self.lower_weight.shape)
            self.lower_weight -= delta_w  # Apply update

        # Update biases if trainable
        if self.train_bias:
            # Update upper_bias
            ind1 = ind2
            ind2 += np.size(self.upper_bias)
            delta_bias = deltas[ind1:ind2].reshape(self.upper_bias.shape)
            self.upper_bias -= delta_bias  # Apply update

            # Update lower_bias
            ind1 = ind2
            ind2 += np.size(self.lower_bias)
            delta_bias = deltas[ind1:ind2].reshape(self.lower_bias.shape)
            self.lower_bias -= delta_bias  # Apply update

        # Update blending factor if trainable
        if self.train_blending:
            ind1 = ind2
            ind2 += np.size(self.blending_factor)
            delta_blend = deltas[ind1:ind2].reshape(self.blending_factor.shape)
            self.blending_factor -= delta_blend  # Apply update

    #################################################################

    def backward(self, error_batch: np.ndarray, learning_rate: float = 1e-3, 
                return_error: bool = False, return_grads: bool = False, modify: bool = True):
        """
        Backward pass for computing gradients and updating parameters.

        Parameters:
        -----------
        error_batch : np.ndarray
            Error propagated from the next layer (batch_size, output_size).
        learning_rate : float, optional
            Rate for updating parameters (default 1e-3).
        return_error : bool, optional
            If True, returns error propagated to inputs.
        return_grads : bool, optional
            If True, returns gradients of parameters.
        modify : bool, optional
            If True, applies updates based on computed gradients.

        Returns:
        --------
        dict or np.ndarray or None
            - {'error_in': error_in, 'gradients': grads} if both `return_error` and `return_grads` are True.
            - `error_in` if `return_error` is True and `return_grads` is False.
            - `gradients` if `return_grads` is True and `return_error` is False.
        """
        # Initialize error to propagate to previous layers if required
        if return_error:
            error_in = np.zeros(self.input.shape)

        # Initialize gradients for weights, biases, and blending factor
        grad_w_up = np.zeros(self.upper_weight.shape) if self.train_weights else None
        grad_w_low = np.zeros(self.lower_weight.shape) if self.train_weights else None
        grad_bias_up = np.zeros(self.upper_bias.shape) if self.train_bias else None
        grad_bias_low = np.zeros(self.lower_bias.shape) if self.train_bias else None
        grad_alpha = np.zeros(self.blending_factor.shape) if self.train_blending else None

        # Iterate over each error in the batch to calculate gradients
        for batch_index, one_batch_error in enumerate(error_batch):
            one_batch_error = one_batch_error.reshape((-1, 1))
            # Calculate gradient of blending factor (alpha) if trainable
            if self.train_blending:
                grad_alpha += one_batch_error * \
                            (self.upper_output[batch_index] - self.lower_output[batch_index])

            # Compute propagated error for upper and lower networks
            e_max = self.blending_factor * one_batch_error
            e_min = (1 - self.blending_factor) * one_batch_error

            # Error allocation based on network outputs and blending factor
            e_upper = e_max * np.logical_not(self.minmax_reverse_stat[batch_index]) + \
                    e_min * self.minmax_reverse_stat[batch_index]
            e_lower = e_min * np.logical_not(self.minmax_reverse_stat[batch_index]) + \
                    e_max * self.minmax_reverse_stat[batch_index]

            # Sensitivity of each network's output w.r.t inputs using activation derivatives
            Fprime_up = net2Fprime(self.upper_net[batch_index], self.activation, self.alpha_activation)
            Fprime_low = net2Fprime(self.lower_net[batch_index], self.activation, self.alpha_activation)

            # Calculate deltas or sensitivities for weights and biases
            sensitivity_up = e_upper * Fprime_up
            sensitivity_low = e_lower * Fprime_low

            # Update gradients for weights if trainable
            if self.train_weights:
                grad_w_up += np.outer(sensitivity_up.ravel(), self.input[batch_index].ravel())
                grad_w_low += np.outer(sensitivity_low.ravel(), self.input[batch_index].ravel())

            # Update gradients for biases if trainable
            if self.train_bias:
                grad_bias_up += sensitivity_up
                grad_bias_low += sensitivity_low

            # Propagate error to previous layer inputs if return_error is True
            if return_error:
                error_in[batch_index] = np.ravel(self.upper_weight.T @ sensitivity_up + self.lower_weight.T @ sensitivity_low)

        # Average gradients over batch size
        if self.train_weights:
            grad_w_up /= error_batch.shape[0]
            grad_w_up += self.L1_coe * np.sign(self.upper_weight) + self.L2_coe * self.upper_weight
            grad_w_low /= error_batch.shape[0]
            grad_w_low += self.L1_coe * np.sign(self.lower_weight) + self.L2_coe * self.lower_weight
        if self.train_bias:
            grad_bias_up /= error_batch.shape[0]
            grad_bias_low /= error_batch.shape[0]
        if self.train_blending:
            grad_alpha /= error_batch.shape[0]

        # Prepare gradients for update if they exist
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1,1))
        if grads is not None:
            if grad_w_up is not None:
                grads = np.concatenate((grads, grad_w_up.reshape((-1,1))))
                grads = np.concatenate((grads, grad_w_low.reshape((-1,1))))
            if grad_bias_up is not None:
                grads = np.concatenate((grads, grad_bias_up.reshape((-1,1))))
                grads = np.concatenate((grads, grad_bias_low.reshape((-1,1))))
            if grad_alpha is not None:
                grads = np.concatenate((grads, grad_alpha.reshape((-1,1))))
        # Apply updates if modify is True
        if modify:
            self.update(grads, learning_rate=learning_rate)

        # Return requested outputs based on function arguments
        if return_error and return_grads:
            return {'error_in': error_in, 'gradients': grads}
        elif return_error:
            return error_in
        elif return_grads:
            return grads

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class TimeRough:
    """
    Implements a neural network layer-like structure with separate upper and lower networks. 
    Includes an additional alpha parameter to control blending between the two networks.

    Parameters:
    ----------
    time_steps : int
        Number of time steps or sequences.
    input_size : int
        Number of input neurons or features.
    output_size : int
        Number of output neurons.
    use_bias : bool, optional
        Whether to use bias in the upper and lower networks (default is True).
    batch_size : int, optional
        Number of samples per batch (default is 32).
    train_weights : bool, optional
        Whether to train weights (default is True).
    train_bias : bool, optional
        Whether to train bias if use_bias is enabled (default is True).
    train_blending : bool, optional
        Whether to train blending factor or weight of the average (default is False).
    activation : str, optional
        Activation function to use (default is 'sigmoid'). Options include 'relu', 'tanh', etc.
    alpha_acti : float, optional
        Scaling parameter for the activation function (default is None).
    weights_uniform_range : tuple, optional
        Range for initializing weights uniformly (default is (-1, 1)).
    L2_coe : float, optional
        L2 regularization coefficient (default is 0.0).
    L1_coe : float, optional
        L1 regularization coefficient (default is 0.0).
    """

    def __init__(self, 
                 time_steps: int, 
                 input_size: int, 
                 output_size: int, 
                 use_bias: bool = True, 
                 batch_size: int = 32,
                 train_weights: bool = True, 
                 train_bias: bool = True, 
                 train_blending: bool = False,
                 activation: str = 'sigmoid', 
                 alpha_acti: float = None, 
                 weights_uniform_range: tuple = (-1, 1),
                 L2_coe: float = 0.0, 
                 L1_coe: float = 0.0) -> None:
        """
        Initializes the TimeRough class with the specified configuration.
        """
        self.output_size = output_size  # Number of output neurons.
        self.input_size = input_size  # Number of input neurons/features.
        self.time_steps = time_steps  # Number of time steps.
        self.batch_size = batch_size  # Batch size.
        self.L2_coe = L2_coe  # L2 regularization coefficient.
        self.L1_coe = L1_coe  # L1 regularization coefficient.
        self.use_bias = use_bias  # Whether to use bias terms.
        self.activation = activation  # Activation function.
        self.alpha_activation = alpha_acti  # Alpha activation scaling parameter.
        self.train_weights = train_weights  # Whether weights are trainable.
        self.train_bias = False if use_bias is False else train_bias  # Bias trainability depends on use_bias.
        self.train_blending = train_blending  # Whether the blending factor is trainable.

        # Compute ranges for initializing upper and lower weights.
        middle = (weights_uniform_range[0] + weights_uniform_range[1]) / 2
        upper_range = (weights_uniform_range[0], middle)  # Range for upper weights.
        lower_range = (middle, weights_uniform_range[1])  # Range for lower weights.

        # Initialize upper and lower weights using a uniform distribution.
        self.upper_weight = Dense_weight_init(input_size, output_size, method="uniform", ranges=upper_range)
        self.lower_weight = Dense_weight_init(input_size, output_size, method="uniform", ranges=lower_range)

        # Initialize biases if use_bias is enabled.
        if self.use_bias:
            self.upper_bias = np.zeros((output_size, 1))  # Bias for the upper network.
            self.lower_bias = np.zeros((output_size, 1))  # Bias for the lower network.

        # Initialize the blending factor (alpha) between the upper and lower networks.
        self.blending_factor = np.zeros((output_size, 1)) + 0.5

        # Allocate memory for intermediate and final network outputs.
        self.upper_net = np.zeros((batch_size, time_steps, output_size, 1))  # Output for the upper network.
        self.lower_net = np.zeros((batch_size, time_steps, output_size, 1))  # Output for the lower network.

        # Track min-max reverse operation results during computation.
        self.minmax_reverse_stat = np.zeros((batch_size, time_steps, output_size))

        # Final and intermediate outputs.
        self.output = np.zeros((batch_size, time_steps, output_size, 1))  # Final output.
        self.upper_output = np.zeros((batch_size, time_steps, output_size, 1))  # Upper network output.
        self.lower_output = np.zeros((batch_size, time_steps, output_size, 1))  # Lower network output.
        self.input = np.zeros((batch_size, time_steps, input_size, 1))  # Input data placeholder.

        # Initialize gradients for training.
        self.grad_w_up = np.zeros(self.upper_weight.shape) if self.train_weights else None  # Gradients for upper weights.
        self.grad_w_low = np.zeros(self.lower_weight.shape) if self.train_weights else None  # Gradients for lower weights.
        self.grad_b_low = np.zeros(self.lower_bias.shape) if self.train_bias else None  # Gradients for lower bias.
        self.grad_b_up = np.zeros(self.upper_bias.shape) if self.train_bias else None  # Gradients for upper bias.
        self.grad_alpha = np.zeros(self.blending_factor.shape) if self.train_blending else None  # Gradients for blending factor.

    #################################################################

    def trainable_params(self) -> int:
        """
        Returns the number of trainable parameters in the model.

        Includes weights, biases (if enabled), and the blending factor (if trainable).

        Returns:
        --------
        int
            Total number of trainable parameters.
        """
        params = 0  # Initialize parameter counter.

        # Count trainable weights (upper and lower).
        if self.train_weights:
            params += np.size(self.upper_weight) * 2

        # Count trainable biases (upper and lower).
        if self.train_bias:
            params += np.size(self.upper_bias) * 2

        # Count trainable blending factor (alpha).
        if self.train_blending:
            params += np.size(self.blending_factor)

        return int(params)  # Ensure return value is an integer.

    #################################################################

    def all_params(self) -> int:
        """
        Returns the total number of parameters in the model.

        Includes all weights, biases (if enabled), and the blending factor.

        Returns:
        --------
        int
            Total number of parameters in the model.
        """
        # Start with weights and blending factor.
        params = np.size(self.upper_weight) * 2 + np.size(self.blending_factor)

        # Add biases if use_bias is enabled.
        if self.use_bias:
            params += np.size(self.upper_bias) * 2

        return int(params)  # Ensure return value is an integer.

    #################################################################

    def __call__(self, batch_index: int, seq_index: int, input: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass of the model.

        Parameters:
        -----------
        batch_index : int
            The index of the current batch.
        seq_index : int
            The index of the current sequence.
        input : np.ndarray
            Input data for the current time step, expected to be a vector.

        Returns:
        --------
        np.ndarray
            The output of the model for the given batch and sequence, as a vector.
        """
        # Store the reshaped input for the current batch and sequence.
        self.input[batch_index, seq_index] = input.reshape((-1, 1))

        # Ensure the batch index does not exceed the configured batch size.
        if self.batch_size < batch_index:
            raise ValueError("Data batch size cannot exceed the model's batch size.")

        # Compute the net input for the upper and lower networks.
        self.upper_net[batch_index, seq_index] = self.upper_weight @ input.reshape((-1, 1))
        self.lower_net[batch_index, seq_index] = self.lower_weight @ input.reshape((-1, 1))

        # Add biases if enabled.
        if self.use_bias:
            self.upper_net[batch_index, seq_index] += self.upper_bias
            self.lower_net[batch_index, seq_index] += self.lower_bias

        # Apply activation functions to the outputs of the upper and lower networks.
        up_out = net2out(self.upper_net[batch_index, seq_index], self.activation, alpha=self.alpha_activation)
        low_out = net2out(self.lower_net[batch_index, seq_index], self.activation, alpha=self.alpha_activation)

        # Concatenate the outputs of the upper and lower networks to compute min and max values.
        concat_out = np.concatenate((up_out, low_out), axis=1)
        self.minmax_reverse_stat[batch_index, seq_index] = np.argmax(concat_out)

        # Compute the max value for the upper network and the min value for the lower network.
        self.upper_output[batch_index, seq_index] = np.max(concat_out, axis=1).reshape((-1, 1))
        self.lower_output[batch_index, seq_index] = np.min(concat_out, axis=1).reshape((-1, 1))

        # Blend the outputs of the upper and lower networks using the blending factor.
        self.output[batch_index, seq_index] = (
            self.blending_factor * self.upper_output[batch_index, seq_index]
            + (1 - self.blending_factor) * self.lower_output[batch_index, seq_index]
        )

        # Return the blended output as the final result for the given batch and sequence.
        return self.output[batch_index, seq_index]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for updating network parameters.

        Parameters:
        -----------
        optimizer : str, optional
            The type of optimizer to use (default is 'Adam').
        **kwargs : dict, optional
            Additional parameters for configuring the optimizer.

        Returns:
        --------
        None
        """
        # Initialize the optimizer with the specified method and additional parameters.
        # The optimizer will handle updates for all trainable parameters in the network.
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, batch_size: int, learning_rate: float = 1e-3, grads: np.ndarray = None) -> None:
        """
        Updates the layer's weights and biases based on calculated gradients.

        Parameters:
        -----------
        batch_size : int
            The batch size used to calculate the average gradients.
        learning_rate : float, optional
            The step size for parameter updates (default is 1e-3).
        grads : np.ndarray, optional
            Precomputed gradients for the parameters. If None, gradients are computed internally.

        Returns:
        --------
        None
        """
        if grads is None:
            # Apply L1 and L2 regularization to the gradients for weights if trainable.
            if self.train_weights:
                self.grad_w_up += self.L1_coe * np.sign(self.upper_weight) + self.L2_coe * self.upper_weight
                self.grad_w_low += self.L1_coe * np.sign(self.lower_weight) + self.L2_coe * self.lower_weight

            # Concatenate all gradients into a single array if needed for updates.
            grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
            if grads is not None:
                if self.grad_w_up is not None:
                    grads = np.concatenate((grads, self.grad_w_up.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_low.reshape((-1, 1))))
                if self.grad_b_up is not None:
                    grads = np.concatenate((grads, self.grad_b_up.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_b_low.reshape((-1, 1))))
                if self.grad_alpha is not None:
                    grads = np.concatenate((grads, self.grad_alpha.reshape((-1, 1))))
                # Normalize the gradients by dividing by the batch size.
                grads /= batch_size

        # Compute parameter update deltas using the optimizer and gradients.
        deltas = self.Optimizer(grads, learning_rate)

        # Initialize an index for iterating through the deltas.
        ind2 = 0

        # Update the upper and lower weights if trainable.
        if self.train_weights:
            ind1 = ind2
            ind2 += int(np.size(self.upper_weight))
            delta_w = deltas[ind1:ind2].reshape(self.upper_weight.shape)
            self.upper_weight -= delta_w

            ind1 = ind2
            ind2 += int(np.size(self.lower_weight))
            delta_w = deltas[ind1:ind2].reshape(self.lower_weight.shape)
            self.lower_weight -= delta_w

        # Update the upper and lower biases if trainable.
        if self.train_bias:
            ind1 = ind2
            ind2 += np.size(self.upper_bias)
            delta_bias = deltas[ind1:ind2].reshape(self.upper_bias.shape)
            self.upper_bias -= delta_bias

            ind1 = ind2
            ind2 += np.size(self.lower_bias)
            delta_bias = deltas[ind1:ind2].reshape(self.lower_bias.shape)
            self.lower_bias -= delta_bias

        # Update the blending factor if trainable.
        if self.train_blending:
            ind1 = ind2
            ind2 += np.size(self.blending_factor)
            delta_blend = deltas[ind1:ind2].reshape(self.blending_factor.shape)
            self.blending_factor -= delta_blend

        # Reset all gradients to zero after updates are applied.
        self.grad_w_up = self.grad_w_up * 0 if self.train_weights else None
        self.grad_w_low = self.grad_w_low * 0 if self.train_weights else None
        self.grad_b_up = self.grad_b_up * 0 if self.train_bias else None
        self.grad_b_low = self.grad_b_low * 0 if self.train_bias else None
        self.grad_alpha = self.grad_alpha * 0 if self.train_blending else None

    #################################################################

    def return_grads(self) -> np.ndarray:
        """
        Collects gradients for all trainable parameters into a single array.

        Returns:
        --------
        np.ndarray
            A single concatenated array containing gradients for all trainable parameters.
            If there are no trainable parameters, returns None.
        """
        # Initialize an empty gradient array or None if there are no trainable parameters.
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))

        if self.grad_w is not None:
            # Add gradients for upper and lower weights if weights are trainable.
            if self.train_weights:
                grad_w_up = self.grad_w_up + (self.L1_coe * np.sign(self.upper_weight) + self.L2_coe * self.upper_weight)
                grad_w_low = self.grad_w_low + (self.L1_coe * np.sign(self.lower_weight) + self.L2_coe * self.lower_weight)
                grads = np.concatenate((grads, grad_w_up.reshape((-1, 1))))
                grads = np.concatenate((grads, grad_w_low.reshape((-1, 1))))

            # Add gradients for biases if biases are trainable.
            if self.grad_b is not None:
                grads = np.concatenate((grads, self.grad_b_up.reshape((-1, 1))))
                grads = np.concatenate((grads, self.grad_b_low.reshape((-1, 1))))

            # Add gradients for blending factors if they are trainable
            if self.train_blending:
                grads = np.concatenate((grads, self.grad_alpha.reshape((-1, 1))))

        # Return the collected gradients.
        return grads

    #################################################################

    def backward(self, batch_index: int, seq_index: int, error: np.ndarray) -> np.ndarray:
        """
        Computes gradients for weights, biases, and optionally updates parameters.
        Propagates errors to the inputs of the previous layer.

        Parameters:
        -----------
        batch_index : int
            Index of the current batch.
        seq_index : int
            Index of the current sequence.
        error : np.ndarray
            Error signal received from the subsequent layer or time step.

        Returns:
        --------
        np.ndarray
            Propagated error for the input, reshaped as a vector.
        """
        # Update the gradient of the blending factor (alpha) if trainable.
        if self.train_blending:
            self.grad_alpha += error.reshape((-1, 1)) * (
                self.upper_output[batch_index, seq_index] - self.lower_output[batch_index, seq_index]
            )

        # Allocate error between upper and lower networks based on blending factor.
        e_max = self.blending_factor * error.reshape((-1, 1))
        e_min = (1 - self.blending_factor) * error.reshape((-1, 1))

        # Split the error allocation based on the network outputs and blending factor.
        e_upper = (
            e_max * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1)))
            + e_min * self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))
        )
        e_lower = (
            e_min * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1)))
            + e_max * self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))
        )

        # Compute activation derivatives for both upper and lower networks.
        Fprime_up = net2Fprime(self.upper_net[batch_index, seq_index], self.activation, self.alpha_activation)
        Fprime_low = net2Fprime(self.lower_net[batch_index, seq_index], self.activation, self.alpha_activation)

        # Calculate sensitivity for weights and biases using allocated errors and activation derivatives.
        sensitivity_up = e_upper.reshape((-1, 1)) * Fprime_up
        sensitivity_low = e_lower.reshape((-1, 1)) * Fprime_low

        # Update gradients for weights if trainable.
        if self.train_weights:
            self.grad_w_up += np.outer(sensitivity_up.ravel(), self.input[batch_index, seq_index].ravel())
            self.grad_w_low += np.outer(sensitivity_low.ravel(), self.input[batch_index, seq_index].ravel())

        # Update gradients for biases if trainable.
        if self.train_bias:
            self.grad_b_up += sensitivity_up
            self.grad_b_low += sensitivity_low

        # Propagate the error backward to the inputs of the previous layer.
        error_in = np.ravel(
            self.upper_weight.T @ sensitivity_up + self.lower_weight.T @ sensitivity_low
        )

        # Return the propagated error as a reshaped vector.
        return error_in.reshape((-1, 1))

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class Rough1Feedback:
    """
    Implements a neural network layer-like structure with separate upper and lower networks.
    Adds a feedback mechanism via state vectors and includes a blend factor for combining outputs.

    Parameters:
    ----------
    time_steps : int
        Number of time steps or sequences.
    input_size : int
        Number of input neurons or features.
    output_size : int
        Number of output neurons.
    feedback_size : int
        Number of rows in the state vector.
    use_bias : bool, optional
        Whether to use bias in the upper and lower networks (default is True).
    batch_size : int, optional
        Number of samples per batch (default is 32).
    train_bias : bool, optional
        Whether to train biases if use_bias is enabled (default is True).
    train_weights : bool, optional
        Whether to train weights (default is True).
    train_blending : bool, optional
        Whether to train the blending factor (default is False).
    activation : str, optional
        Activation function to use (default is 'sigmoid').
    alpha_acti : float, optional
        Scaling parameter for the activation function (default is None).
    weights_uniform_range : tuple, optional
        Range for initializing weights uniformly (default is (-1, 1)).
    L2_coe : float, optional
        L2 regularization coefficient (default is 0.0).
    L1_coe : float, optional
        L1 regularization coefficient (default is 0.0).

    Attributes:
    ----------
    upper_weight : np.ndarray
        Weight matrix for the upper network (input to output).
    lower_weight : np.ndarray
        Weight matrix for the lower network (input to output).
    upper_weight_state : np.ndarray
        Weight matrix for the upper network's state feedback.
    lower_weight_state : np.ndarray
        Weight matrix for the lower network's state feedback.
    upper_bias : np.ndarray
        Bias vector for the upper network.
    lower_bias : np.ndarray
        Bias vector for the lower network.
    blending_factor : np.ndarray
        Blending factor to combine outputs from the upper and lower networks.
    """

    def __init__(self, 
                 time_steps: int, 
                 input_size: int, 
                 output_size: int, 
                 feedback_size: int = None,
                 use_bias: bool = True, 
                 batch_size: int = 32, 
                 train_weights: bool = True, 
                 train_bias: bool = True,
                 train_blending: bool = False, 
                 activation: str = 'sigmoid', 
                 alpha_acti: float = None,
                 weights_uniform_range: tuple = (-1, 1), 
                 L2_coe: float = 0.0, 
                 L1_coe: float = 0.0) -> None:
        """
        Initializes the Rough1Feedback class with the specified configuration.

        Parameters:
        ----------
        time_steps : int
            Number of time steps or sequences.
        input_size : int
            Number of input neurons or features.
        output_size : int
            Number of output neurons.
        feedback_size : int, optional
            Number of rows in the state vector (default is equal to output_size).
        use_bias : bool, optional
            Whether to use bias in the networks (default is True).
        batch_size : int, optional
            Number of samples per batch (default is 32).
        train_weights : bool, optional
            Whether to train weights (default is True).
        train_bias : bool, optional
            Whether to train biases if use_bias is enabled (default is True).
        train_blending : bool, optional
            Whether to train the blending factor (default is False).
        activation : str, optional
            Activation function to use (default is 'sigmoid').
        alpha_acti : float, optional
            Scaling parameter for the activation function (default is None).
        weights_uniform_range : tuple, optional
            Range for initializing weights uniformly (default is (-1, 1)).
        L2_coe : float, optional
            L2 regularization coefficient (default is 0.0).
        L1_coe : float, optional
            L1 regularization coefficient (default is 0.0).
        """
        self.output_size = output_size
        self.input_size = input_size
        self.feedback_size = feedback_size if feedback_size is not None else output_size
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.L2_coe = L2_coe
        self.L1_coe = L1_coe
        self.use_bias = use_bias
        self.activation = activation
        self.alpha_activation = alpha_acti
        self.train_weights = train_weights
        self.train_bias = False if use_bias is False else train_bias
        self.train_blending = train_blending

        # Compute ranges for weight initialization.
        middle = (weights_uniform_range[0] + weights_uniform_range[1]) / 2
        upper_range = (weights_uniform_range[0], middle)
        lower_range = (middle, weights_uniform_range[1])

        # Initialize weights for input connections.
        self.upper_weight = Dense_weight_init(input_size, output_size, method="uniform", ranges=upper_range)
        self.lower_weight = Dense_weight_init(input_size, output_size, method="uniform", ranges=lower_range)

        # Initialize weights for state feedback connections.
        self.lower_weight_state = Dense_weight_init(self.feedback_size, output_size, method="uniform", ranges=lower_range)
        self.upper_weight_state = Dense_weight_init(self.feedback_size, output_size, method="uniform", ranges=upper_range)

        # Initialize biases if enabled.
        if self.use_bias:
            self.upper_bias = np.zeros((output_size, 1))
            self.lower_bias = np.zeros((output_size, 1))

        # Initialize the blending factor (alpha) to combine upper and lower network outputs.
        self.blending_factor = np.zeros((output_size, 1)) + 0.5

        # Allocate memory for intermediate and final outputs.
        self.upper_net = np.zeros((batch_size, time_steps, output_size, 1))
        self.lower_net = np.zeros((batch_size, time_steps, output_size, 1))
        self.output = np.zeros((batch_size, time_steps, output_size, 1))
        self.upper_output = np.zeros((batch_size, time_steps, output_size, 1))
        self.lower_output = np.zeros((batch_size, time_steps, output_size, 1))
        self.input = np.zeros((batch_size, time_steps, input_size, 1))

        # Store min-max reverse operation results during computation.
        self.minmax_reverse_stat = np.zeros((batch_size, time_steps, output_size))

        # Initialize gradients for training.
        self.grad_w_up = np.zeros(self.upper_weight.shape) if self.train_weights else None
        self.grad_w_low = np.zeros(self.lower_weight.shape) if self.train_weights else None
        self.grad_w_up_state = np.zeros(self.upper_weight_state.shape) if self.train_weights else None
        self.grad_w_low_state = np.zeros(self.lower_weight_state.shape) if self.train_weights else None
        self.grad_b_low = np.zeros(self.lower_bias.shape) if self.train_bias else None
        self.grad_b_up = np.zeros(self.upper_bias.shape) if self.train_bias else None
        self.grad_alpha = np.zeros(self.blending_factor.shape) if self.train_blending else None

    #################################################################

    def trainable_params(self) -> int:
        """
        Returns the number of trainable parameters in the model.

        Includes weights, biases (if enabled), and the blending factor (if trainable).

        Returns:
        --------
        int
            Total number of trainable parameters.
        """
        params = 0
        if self.train_weights:
            params += np.size(self.upper_weight) * 2  # Includes both upper and lower weights.
            params += np.size(self.upper_weight_state) * 2  # Includes weights for state feedback.
        if self.train_bias:
            params += np.size(self.upper_bias) * 2  # Includes biases for both upper and lower networks.
        if self.train_blending:
            params += np.size(self.blending_factor)  # Includes blending factor (alpha).
        return int(params)

    #################################################################

    def all_params(self) -> int:
        """
        Returns the total number of parameters in the model.

        Includes all weights, biases (if enabled), and the blending factor.

        Returns:
        --------
        int
            Total number of parameters in the model.
        """
        params = (np.size(self.upper_weight) + np.size(self.upper_weight_state)) * 2
        params += np.size(self.blending_factor)
        if self.use_bias:
            params += np.size(self.upper_bias) * 2  # Includes biases for both upper and lower networks.
        return int(params)

    #################################################################

    def __call__(self, batch_index: int, seq_index: int, input: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass of the model.

        Parameters:
        -----------
        batch_index : int
            Index of the current batch being processed.
        seq_index : int
            Index of the current sequence in the batch.
        input : np.ndarray
            Input data for the current time step, shaped as a vector.
        state : np.ndarray
            State data from the feedback mechanism, shaped as a vector.

        Returns:
        --------
        np.ndarray
            The output of the model for the current time step, shaped as a vector.
        """
        # Reshape input and state to ensure proper dimensionality for computations.
        self.input[batch_index, seq_index] = input.reshape((-1, 1))
        state = state.reshape((-1, 1))

        # Validate batch size to ensure the batch index is within the defined range.
        if self.batch_size < batch_index:
            raise ValueError('Data batch size cannot be larger than model batch size.')

        # Compute the net inputs for both the upper and lower networks.
        # For the upper network, include contributions from input weights and state feedback weights.
        self.upper_net[batch_index, seq_index] = (
            self.upper_weight @ input.reshape((-1, 1))
            + self.upper_weight_state @ state.reshape((-1, 1))
        )
        # For the lower network, include contributions from input weights and state feedback weights.
        self.lower_net[batch_index, seq_index] = (
            self.lower_weight @ input.reshape((-1, 1))
            + self.lower_weight_state @ state.reshape((-1, 1))
        )

        # Add biases to the net inputs if biases are enabled.
        if self.use_bias:
            self.upper_net[batch_index, seq_index] += self.upper_bias
            self.lower_net[batch_index, seq_index] += self.lower_bias

        # Apply the specified activation function to compute outputs from the net inputs.
        up_out = net2out(self.upper_net[batch_index, seq_index], self.activation, alpha=self.alpha_activation)
        low_out = net2out(self.lower_net[batch_index, seq_index], self.activation, alpha=self.alpha_activation)

        # Combine outputs from the upper and lower networks.
        # Concatenate the outputs along the second dimension for min-max operations.
        concat_out = np.concatenate((up_out, low_out), axis=1)

        # Record the index of the maximum value in each output dimension for reverse operations.
        self.minmax_reverse_stat[batch_index, seq_index] = np.argmax(concat_out)

        # Compute the maximum (upper) and minimum (lower) outputs for blending.
        self.upper_output[batch_index, seq_index] = np.max(concat_out, axis=1).reshape((-1, 1))
        self.lower_output[batch_index, seq_index] = np.min(concat_out, axis=1).reshape((-1, 1))

        # Compute the final blended output using the blending factor (alpha).
        self.output[batch_index, seq_index] = (
            self.blending_factor * self.upper_output[batch_index, seq_index]
            + (1 - self.blending_factor) * self.lower_output[batch_index, seq_index]
        )

        # Return the final output for the current time step.
        return self.output[batch_index, seq_index]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for updating network parameters.

        Parameters:
        -----------
        optimizer : str, optional
            Specifies the type of optimizer to use. Default is 'Adam'.
            Other options include 'SGD', 'RMSprop', etc.
        **kwargs : dict, optional
            Additional configuration parameters for the chosen optimizer.
            For example, learning rate, momentum, etc.

        Returns:
        --------
        None
        """
        # Create and configure the optimizer with the specified method and parameters.
        # The optimizer will handle updates to all trainable parameters in the network.
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, batch_size: int, learning_rate: float = 1e-3, grads: np.ndarray = None) -> None:
        """
        Updates the layer's weights, biases, and blending factor based on calculated gradients.

        Parameters:
        -----------
        batch_size : int
            The size of the batch used during training to average the gradients.
        learning_rate : float, optional
            The step size for parameter updates in each iteration (default is 1e-3).
        grads : np.ndarray, optional
            Pre-computed gradients for the parameters. If not provided, gradients
            will be calculated based on the internal states of the model.

        Returns:
        --------
        None
        """
        if grads is None:
            # Add L1 and L2 regularization contributions to the gradients for weights.
            if self.train_weights:
                # Regularize and calculate gradients for weights and state feedback weights.
                self.grad_w_up += (self.L1_coe * np.sign(self.upper_weight) + self.L2_coe * self.upper_weight)
                self.grad_w_low += (self.L1_coe * np.sign(self.lower_weight) + self.L2_coe * self.lower_weight)
                self.grad_w_up_state += (self.L1_coe * np.sign(self.upper_weight_state) + self.L2_coe * self.upper_weight_state)
                self.grad_w_low_state += (self.L1_coe * np.sign(self.lower_weight_state) + self.L2_coe * self.lower_weight_state)

            # Combine gradients into a single array for parameter updates.
            grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
            if grads is not None:
                # Append gradients for upper and lower weights and their state weights.
                if self.grad_w_up is not None:
                    grads = np.concatenate((grads, self.grad_w_up.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_low.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_up_state.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_low_state.reshape((-1, 1))))
                # Append gradients for upper and lower biases if applicable.
                if self.grad_b_up is not None:
                    grads = np.concatenate((grads, self.grad_b_up.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_b_low.reshape((-1, 1))))
                # Append gradients for blending factor if trainable.
                if self.grad_alpha is not None:
                    grads = np.concatenate((grads, self.grad_alpha.reshape((-1, 1))))
                # Average gradients across the batch.
                grads /= batch_size

        # Calculate updates for each parameter using the optimizer and provided gradients.
        deltas = self.Optimizer(grads, learning_rate)

        # Initialize an index to track parameter positions within deltas.
        ind2 = 0

        # Update weights if trainable.
        if self.train_weights:
            # Update upper weights for input connections.
            ind1 = ind2
            ind2 += int(np.size(self.upper_weight))
            delta_w = deltas[ind1:ind2].reshape(self.upper_weight.shape)
            self.upper_weight -= delta_w  # Apply update.

            # Update lower weights for input connections.
            ind1 = ind2
            ind2 += int(np.size(self.lower_weight))
            delta_w = deltas[ind1:ind2].reshape(self.lower_weight.shape)
            self.lower_weight -= delta_w  # Apply update.

            # Update upper weights for state feedback connections.
            ind1 = ind2
            ind2 += int(np.size(self.upper_weight_state))
            delta_w = deltas[ind1:ind2].reshape(self.upper_weight_state.shape)
            self.upper_weight_state -= delta_w  # Apply update.

            # Update lower weights for state feedback connections.
            ind1 = ind2
            ind2 += int(np.size(self.lower_weight_state))
            delta_w = deltas[ind1:ind2].reshape(self.lower_weight_state.shape)
            self.lower_weight_state -= delta_w  # Apply update.

        # Update biases if trainable.
        if self.train_bias:
            # Update upper biases.
            ind1 = ind2
            ind2 += np.size(self.upper_bias)
            delta_bias = deltas[ind1:ind2].reshape(self.upper_bias.shape)
            self.upper_bias -= delta_bias  # Apply update.

            # Update lower biases.
            ind1 = ind2
            ind2 += np.size(self.lower_bias)
            delta_bias = deltas[ind1:ind2].reshape(self.lower_bias.shape)
            self.lower_bias -= delta_bias  # Apply update.

        # Update blending factor if trainable.
        if self.train_blending:
            ind1 = ind2
            ind2 += np.size(self.blending_factor)
            delta_blend = deltas[ind1:ind2].reshape(self.blending_factor.shape)
            self.blending_factor -= delta_blend  # Apply update.

        # Reset gradients to zero for the next iteration.
        self.grad_w_up = self.grad_w_up * 0 if self.train_weights else None
        self.grad_w_low = self.grad_w_low * 0 if self.train_weights else None
        self.grad_w_up_state = self.grad_w_up_state * 0 if self.train_weights else None
        self.grad_w_low_state = self.grad_w_low_state * 0 if self.train_weights else None
        self.grad_b_up = self.grad_b_up * 0 if self.train_bias else None
        self.grad_b_low = self.grad_b_low * 0 if self.train_bias else None
        self.grad_alpha = self.grad_alpha * 0 if self.train_blending else None

    #################################################################

    def return_grads(self) -> np.ndarray:
        """
        Aggregates and returns the computed gradients for all trainable parameters in the model.

        Returns:
        --------
        np.ndarray:
            A single array containing all concatenated gradients for weights, biases, and alpha (if trainable).
            If there are no trainable parameters, returns an empty array.
        """
        # Initialize gradients container, starting as an empty array if no trainable parameters.
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))

        if self.grad_w_up is not None:
            # Compute gradients for upper and lower weights, including L1 and L2 regularization.
            grad_w_up = self.grad_w_up + (self.L1_coe * np.sign(self.upper_weight) + self.L2_coe * self.upper_weight)
            grad_w_low = self.grad_w_low + (self.L1_coe * np.sign(self.lower_weight) + self.L2_coe * self.lower_weight)
            grad_w_up_state = self.grad_w_up_state + (
                self.L1_coe * np.sign(self.upper_weight_state) + self.L2_coe * self.upper_weight_state
            )
            grad_w_low_state = self.grad_w_low_state + (
                self.L1_coe * np.sign(self.lower_weight_state) + self.L2_coe * self.lower_weight_state
            )
            # Concatenate computed gradients for input and state weights.
            grads = np.concatenate((grads, grad_w_up.reshape((-1, 1))))
            grads = np.concatenate((grads, grad_w_low.reshape((-1, 1))))
            grads = np.concatenate((grads, grad_w_up_state.reshape((-1, 1))))
            grads = np.concatenate((grads, grad_w_low_state.reshape((-1, 1))))

        if self.grad_b is not None:
            # Concatenate computed gradients for biases, if applicable.
            grads = np.concatenate((grads, self.grad_b.reshape((-1, 1))))

        return grads

    #################################################################

    def backward(self, batch_index: int, seq_index: int, error: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes gradients for weights and biases during backpropagation and propagates the error to inputs.

        Parameters:
        -----------
        batch_index : int
            The current batch index being processed.
        seq_index : int
            The current sequence index within the batch.
        error : np.ndarray
            The error signal from the subsequent layer or time step, shaped as a vector.
        state : np.ndarray
            The state vector used for calculating gradients for state-dependent weights.

        Returns:
        --------
        tuple[np.ndarray, np.ndarray]:
            - The propagated error to the inputs, shaped as a vector.
            - The propagated error to the state, shaped as a vector.
        """
        # Compute gradient of the blending factor (alpha) if trainable.
        if self.train_blending:
            self.grad_alpha += error.reshape((-1, 1)) * (
                self.upper_output[batch_index, seq_index] - self.lower_output[batch_index, seq_index]
            )

        # Compute the propagated error components for upper and lower networks.
        e_max = self.blending_factor * error.reshape((-1, 1))
        e_min = (1 - self.blending_factor) * error.reshape((-1, 1))

        # Allocate error contributions based on network outputs and blending factor.
        e_upper = e_max * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))) + \
                e_min * self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))
        e_lower = e_min * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))) + \
                e_max * self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))

        # Compute sensitivity (delta) of each network's output with respect to inputs using activation derivatives.
        Fprime_up = net2Fprime(self.upper_net[batch_index, seq_index], self.activation, self.alpha_activation)
        Fprime_low = net2Fprime(self.lower_net[batch_index, seq_index], self.activation, self.alpha_activation)

        # Compute the sensitivities for the upper and lower networks.
        sensitivity_up = e_upper.reshape((-1, 1)) * Fprime_up
        sensitivity_low = e_lower.reshape((-1, 1)) * Fprime_low

        # Update gradients for weights if trainable.
        if self.train_weights:
            self.grad_w_up += np.outer(sensitivity_up.ravel(), self.input[batch_index, seq_index].ravel())
            self.grad_w_low += np.outer(sensitivity_low.ravel(), self.input[batch_index, seq_index].ravel())
            self.grad_w_up_state += np.outer(sensitivity_up.ravel(), state.ravel())
            self.grad_w_low_state += np.outer(sensitivity_low.ravel(), state.ravel())

        # Update gradients for biases if trainable.
        if self.train_bias:
            self.grad_b_up += sensitivity_up
            self.grad_b_low += sensitivity_low

        # Compute the propagated errors to the previous layer's inputs and states.
        error_in = np.ravel(self.upper_weight.T @ sensitivity_up + self.lower_weight.T @ sensitivity_low)
        error_state = np.ravel(self.upper_weight_state.T @ sensitivity_up + self.lower_weight_state.T @ sensitivity_low)

        return error_in.reshape((-1, 1)), error_state.reshape((-1, 1))

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class Rough2Feedback:
    """
    Implements a neural network layer-like structure with separate weight matrices and biases 
    for upper and lower networks. It also incorporates feedback mechanisms using Elman and Jordan 
    feedback networks, along with a trainable alpha parameter to control the blending of outputs.

    Parameters:
    ----------
    time_steps : int
        Number of time steps or sequences in the input data.
    input_size : int
        The number of input neurons or features.
    output_size : int
        The number of output neurons.
    feedback_size_jordan : int
        The number of elements in the Jordan feedback vector.
    use_bias : bool, optional
        Whether to use bias in the upper and lower networks (default is True).
    batch_size : int, optional
        The number of samples per batch (default is 32).
    train_bias: bool, optional
        Whether to train bias parameters if `use_bias` is True (default is True).
    train_weights: bool, optional
        Whether to train weight matrices (default is True).
    train_blending : bool, optional
        Whether to train the blending factor for combining upper and lower outputs (default is False).
    activation : str, optional
        Activation function to use (default is 'sigmoid'). Other options include 'relu', 'tanh', etc.
    alpha_acti : float, optional
        A scaling parameter for the activation function (default is None).
    weights_uniform_range : tuple, optional
        Range for initializing weights uniformly (default is (-1, 1)).
    L2_coe : float, optional
        Coefficient for L2 regularization (default is 0.0).
    L1_coe : float, optional
        Coefficient for L1 regularization (default is 0.0).
    """

    def __init__(self, time_steps: int, input_size: int, output_size: int, feedback_size_jordan: int,
                 use_bias: bool = True, batch_size: int = 32, train_weights: bool = True,
                 train_bias: bool = True, train_blending: bool = False,
                 activation: str = 'sigmoid', alpha_acti: float = None,
                 weights_uniform_range: tuple = (-1, 1), L2_coe: float = 0.0, L1_coe: float = 0.0):

        self.output_size = output_size  # Number of output neurons
        self.input_size = input_size  # Number of input neurons/features
        self.feedback_size_elman = output_size  # Number of elements in the Elman feedback vector
        self.feedback_size_jordan = feedback_size_jordan  # Number of elements in the Jordan feedback vector
        self.time_steps = time_steps  # Number of time steps
        self.batch_size = batch_size  # Batch size
        self.L2_coe = L2_coe  # L2 regularization coefficient
        self.L1_coe = L1_coe  # L1 regularization coefficient
        self.use_bias = use_bias  # Whether to use bias
        self.activation = activation  # Activation function
        self.alpha_activation = alpha_acti  # Alpha activation scaling parameter
        self.train_weights = train_weights  # Flag to train weights
        self.train_bias = train_bias if use_bias else False  # Train bias only if bias is enabled
        self.train_blending = train_blending  # Flag to train blending factor

        # Split weight initialization ranges into upper and lower halves
        middle = (weights_uniform_range[0] + weights_uniform_range[1]) / 2
        upper_range = (weights_uniform_range[0], middle)
        lower_range = (middle, weights_uniform_range[1])

        # Initialize weights for input, Elman, and Jordan networks using uniform distribution
        self.upper_weight = Dense_weight_init(input_size, output_size, method="uniform", ranges=upper_range)
        self.lower_weight = Dense_weight_init(input_size, output_size, method="uniform", ranges=lower_range)
        self.lower_weight_elman = Dense_weight_init(output_size, output_size, method="uniform", ranges=lower_range)
        self.upper_weight_elman = Dense_weight_init(output_size, output_size, method="uniform", ranges=upper_range)
        self.lower_weight_jordan = Dense_weight_init(feedback_size_jordan, output_size, method="uniform", ranges=lower_range)
        self.upper_weight_jordan = Dense_weight_init(feedback_size_jordan, output_size, method="uniform", ranges=upper_range)

        # Initialize biases if applicable
        if self.use_bias:
            self.upper_bias = np.zeros((output_size, 1))  # Upper bias
            self.lower_bias = np.zeros((output_size, 1))  # Lower bias

        # Initialize blending factor between upper and lower networks
        self.blending_factor = np.zeros((output_size, 1)) + 0.5

        # Initialize intermediate and final network outputs
        self.upper_net = np.zeros((batch_size, time_steps, output_size, 1))
        self.lower_net = np.zeros((batch_size, time_steps, output_size, 1))
        self.minmax_reverse_stat = np.zeros((batch_size, time_steps, output_size))
        self.output = np.zeros((batch_size, time_steps, output_size, 1))
        self.upper_output = np.zeros((batch_size, time_steps, output_size, 1))
        self.lower_output = np.zeros((batch_size, time_steps, output_size, 1))
        self.input = np.zeros((batch_size, time_steps, input_size, 1))

        # Initialize gradients
        self.grad_w_up = np.zeros(self.upper_weight.shape) if self.train_weights else None
        self.grad_w_low = np.zeros(self.lower_weight.shape) if self.train_weights else None
        self.grad_w_up_elman = np.zeros(self.upper_weight_elman.shape) if self.train_weights else None
        self.grad_w_low_elman = np.zeros(self.lower_weight_elman.shape) if self.train_weights else None
        self.grad_w_up_jordan = np.zeros(self.upper_weight_jordan.shape) if self.train_weights else None
        self.grad_w_low_jordan = np.zeros(self.lower_weight_jordan.shape) if self.train_weights else None
        self.grad_b_low = np.zeros(self.lower_bias.shape) if self.train_bias else None
        self.grad_b_up = np.zeros(self.upper_bias.shape) if self.train_bias else None
        self.grad_alpha = np.zeros(self.blending_factor.shape) if self.train_blending else None

    #################################################################

    def trainable_params(self) -> int:
        """
        Calculate and return the total number of trainable parameters in the model.
        
        Returns:
        --------
        int:
            Total number of trainable parameters (weights, biases, and blending factors if applicable).
        """
        params = 0
        if self.train_weights:
            params += np.size(self.upper_weight) * 2  # Upper and lower weights for inputs
            params += np.size(self.upper_weight_elman) * 2  # Upper and lower weights for Elman feedback
            params += np.size(self.upper_weight_jordan) * 2  # Upper and lower weights for Jordan feedback
        if self.train_bias:
            params += np.size(self.upper_bias) * 2  # Biases for upper and lower networks
        if self.train_blending:
            params += np.size(self.blending_factor)  # Blending factor

        return params

    #################################################################

    def all_params(self) -> int:
        """
        Calculate and return the total number of parameters in the model.
        
        Returns:
        --------
        int:
            Total number of parameters (weights, biases, and blending factors if applicable).
        """
        params = (np.size(self.upper_weight) + np.size(self.upper_weight_elman) + np.size(self.upper_weight_jordan)) \
                 * 2 + np.size(self.blending_factor)
        if self.use_bias:
            params += np.size(self.upper_bias) * 2  # Biases for upper and lower networks

        return params

    #################################################################

    def __call__(self, batch_index: int, seq_index: int, input: np.ndarray,
                elman_state: np.ndarray, jordan_state: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass of the model.
        
        Parameters:
        -----------
        batch_index : int
            The current batch index.
        seq_index : int
            The current sequence index.
        input : np.ndarray
            The input data for this batch and sequence, shape should be a vector.
        elman_state : np.ndarray
            The Elman feedback state, shape should be a vector.
        jordan_state : np.ndarray
            The Jordan feedback state, shape should be a vector.
        
        Returns:
        --------
        np.ndarray
            The output of the model for this batch and sequence, shape will be a vector.
        """
        # Reshape inputs and states to ensure correct dimensionality
        self.input[batch_index, seq_index] = input.reshape((-1, 1))
        jordan_state = jordan_state.reshape((-1, 1))
        elman_state = elman_state.reshape((-1, 1))

        # Validate batch size
        if self.batch_size < batch_index:
            raise ValueError('Data batch size cannot be larger than model batch size')

        # Compute net input for the upper network using input, Elman, and Jordan weights
        self.upper_net[batch_index, seq_index] = (
            self.upper_weight @ input.reshape((-1, 1)) +
            self.upper_weight_elman @ elman_state +
            self.upper_weight_jordan @ jordan_state
        )

        # Compute net input for the lower network using input, Elman, and Jordan weights
        self.lower_net[batch_index, seq_index] = (
            self.lower_weight @ input.reshape((-1, 1)) +
            self.lower_weight_elman @ elman_state +
            self.lower_weight_jordan @ jordan_state
        )

        # Add bias terms if applicable
        if self.use_bias:
            self.upper_net[batch_index, seq_index] += self.upper_bias
            self.lower_net[batch_index, seq_index] += self.lower_bias

        # Apply activation function to upper and lower net inputs
        up_out = net2out(self.upper_net[batch_index, seq_index], self.activation, alpha=self.alpha_activation)
        low_out = net2out(self.lower_net[batch_index, seq_index], self.activation, alpha=self.alpha_activation)

        # Concatenate upper and lower outputs to compute max and min values
        concat_out = np.concatenate((up_out, low_out), axis=1)
        self.minmax_reverse_stat[batch_index, seq_index] = np.argmax(concat_out, axis=1)

        # Compute max (upper output) and min (lower output)
        self.upper_output[batch_index, seq_index] = np.max(concat_out, axis=1).reshape((-1, 1))
        self.lower_output[batch_index, seq_index] = np.min(concat_out, axis=1).reshape((-1, 1))

        # Compute the final output by blending upper and lower outputs using the blending factor
        self.output[batch_index, seq_index] = (
            self.blending_factor * self.upper_output[batch_index, seq_index] +
            (1 - self.blending_factor) * self.lower_output[batch_index, seq_index]
        )

        # Return the computed output
        return self.output[batch_index, seq_index]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for updating the network parameters.

        Parameters:
        -----------
        optimizer : str, optional
            The optimization algorithm to use (default is 'Adam').
        **kwargs : dict, optional
            Additional parameters for configuring the optimizer.
        
        Returns:
        --------
        None
        """
        # Initialize the optimizer using the specified method and configuration
        # 'self.trainable_params()' returns the total number of trainable parameters
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, batch_size: int, learning_rate: float = 1e-3, grads: np.ndarray = None) -> None:
        """
        Updates the layer's weights and biases based on the calculated gradients.

        Parameters:
        -----------
        batch_size : int
            The data batch size used to calculate the average of gradients.
        learning_rate : float, optional
            The learning rate for the optimizer (default is 1e-3).
        grads : np.ndarray, optional
            The pre-computed gradients with respect to the parameters. If None, gradients 
            will be internally computed and applied.
        
        Returns:
        --------
        None
        """
        if grads is None:
            # Apply L1 and L2 regularization to the gradients of weights
            if self.train_weights:
                self.grad_w_up += (self.L1_coe * np.sign(self.upper_weight) + self.L2_coe * self.upper_weight)
                self.grad_w_low += (self.L1_coe * np.sign(self.lower_weight) + self.L2_coe * self.lower_weight)
                self.grad_w_up_elman += (self.L1_coe * np.sign(self.upper_weight_elman) + self.L2_coe * self.upper_weight_elman)
                self.grad_w_low_elman += (self.L1_coe * np.sign(self.lower_weight_elman) + self.L2_coe * self.lower_weight_elman)
                self.grad_w_up_jordan += (self.L1_coe * np.sign(self.upper_weight_jordan) + self.L2_coe * self.upper_weight_jordan)
                self.grad_w_low_jordan += (self.L1_coe * np.sign(self.lower_weight_jordan) + self.L2_coe * self.lower_weight_jordan)

            # Initialize an empty array for concatenating gradients
            grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))

            # Concatenate gradients for all trainable parameters
            if grads is not None:
                if self.grad_w_up is not None:
                    grads = np.concatenate((grads, self.grad_w_up.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_low.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_up_elman.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_low_elman.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_up_jordan.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_low_jordan.reshape((-1, 1))))
                if self.grad_b_up is not None:
                    grads = np.concatenate((grads, self.grad_b_up.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_b_low.reshape((-1, 1))))
                if self.grad_alpha is not None:
                    grads = np.concatenate((grads, self.grad_alpha.reshape((-1, 1))))
                grads /= batch_size  # Average gradients over the batch size

        # Compute parameter updates using the optimizer
        deltas = self.Optimizer(grads, learning_rate)

        # Initialize an index for tracking parameter positions in deltas
        ind2 = 0

        # Update weights and biases based on computed deltas
        if self.train_weights:
            # Update upper and lower weights for inputs, Elman, and Jordan networks
            for weight, grad in [
                (self.upper_weight, self.grad_w_up),
                (self.lower_weight, self.grad_w_low),
                (self.upper_weight_elman, self.grad_w_up_elman),
                (self.lower_weight_elman, self.grad_w_low_elman),
                (self.upper_weight_jordan, self.grad_w_up_jordan),
                (self.lower_weight_jordan, self.grad_w_low_jordan),
            ]:
                ind1 = ind2
                ind2 += weight.size
                delta_w = deltas[ind1:ind2].reshape(weight.shape)
                weight -= delta_w  # Apply weight updates

        if self.train_bias:
            # Update biases for upper and lower networks
            for bias, grad in [(self.upper_bias, self.grad_b_up), (self.lower_bias, self.grad_b_low)]:
                ind1 = ind2
                ind2 += bias.size
                delta_b = deltas[ind1:ind2].reshape(bias.shape)
                bias -= delta_b  # Apply bias updates

        if self.train_blending:
            # Update blending factor
            ind1 = ind2
            ind2 += self.blending_factor.size
            delta_blend = deltas[ind1:ind2].reshape(self.blending_factor.shape)
            self.blending_factor -= delta_blend  # Apply blending updates

        # Reset gradients to zero for the next update step
        for grad in [
            self.grad_w_up, self.grad_w_low, self.grad_w_up_elman, self.grad_w_low_elman,
            self.grad_w_up_jordan, self.grad_w_low_jordan, self.grad_b_up, self.grad_b_low, self.grad_alpha
        ]:
            if grad is not None:
                grad *= 0

    #################################################################

    def return_grads(self) -> np.ndarray:
        """
        Compute and return the gradients for all trainable parameters.

        Returns:
        --------
        np.ndarray
            A concatenated array of all gradients, including those for weights, biases, and blending factor,
            if they are trainable.
        """
        # Initialize an empty array to hold the gradients if there are trainable parameters
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
        
        if self.grad_w_up is not None:
            # Apply L1 and L2 regularization for the gradients of weights
            grad_w_up = self.grad_w_up + (self.L1_coe * np.sign(self.upper_weight) + self.L2_coe * self.upper_weight)
            grad_w_low = self.grad_w_low + (self.L1_coe * np.sign(self.lower_weight) + self.L2_coe * self.lower_weight)
            grad_w_up_elman = self.grad_w_up_elman + (self.L1_coe * np.sign(self.upper_weight_elman) + self.L2_coe * self.upper_weight_elman)
            grad_w_low_elman = self.grad_w_low_elman + (self.L1_coe * np.sign(self.lower_weight_elman) + self.L2_coe * self.lower_weight_elman)
            grad_w_up_jordan = self.grad_w_up_jordan + (self.L1_coe * np.sign(self.upper_weight_jordan) + self.L2_coe * self.upper_weight_jordan)
            grad_w_low_jordan = self.grad_w_low_jordan + (self.L1_coe * np.sign(self.lower_weight_jordan) + self.L2_coe * self.lower_weight_jordan)

            # Concatenate all weight gradients
            grads = np.concatenate((grads, grad_w_up.reshape((-1, 1))))
            grads = np.concatenate((grads, grad_w_low.reshape((-1, 1))))
            grads = np.concatenate((grads, grad_w_up_elman.reshape((-1, 1))))
            grads = np.concatenate((grads, grad_w_low_elman.reshape((-1, 1))))
            grads = np.concatenate((grads, grad_w_up_jordan.reshape((-1, 1))))
            grads = np.concatenate((grads, grad_w_low_jordan.reshape((-1, 1))))

        if self.grad_b is not None:
            # Concatenate bias gradients
            grads = np.concatenate((grads, self.grad_b.reshape((-1, 1))))

        return grads

    #################################################################

    def backward(self, batch_index: int, seq_index: int, error: np.ndarray, elman_state: np.ndarray, jordan_state: np.ndarray)\
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients for weights and biases, and propagate errors to previous layers.

        Parameters:
        -----------
        batch_index : int
            The current batch index.
        seq_index : int
            The current sequence index.
        error : np.ndarray
            The error from the subsequent layer or time step.
        elman_state : np.ndarray
            The Elman state vector from the current layer.
        jordan_state : np.ndarray
            The Jordan state vector from the previous layer.

        Returns:
        --------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            - error_in: Error propagated back to the inputs.
            - error_elman: Error propagated to the Elman state.
            - error_jordan: Error propagated to the Jordan state.
        """
        # Update the gradient for the blending factor (if trainable)
        if self.train_blending:
            self.grad_alpha += error.reshape((-1, 1)) * (self.upper_output[batch_index, seq_index] - self.lower_output[batch_index, seq_index])

        # Compute propagated errors for the upper and lower networks
        e_max = self.blending_factor * error.reshape((-1, 1))
        e_min = (1 - self.blending_factor) * error.reshape((-1, 1))

        # Allocate errors based on the blending factor and reverse statistics
        e_upper = e_max * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))) + e_min *\
            self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))
        e_lower = e_min * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))) + e_max *\
            self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))

        # Compute activation derivatives for the upper and lower networks
        Fprime_up = net2Fprime(self.upper_net[batch_index, seq_index], self.activation, self.alpha_activation)
        Fprime_low = net2Fprime(self.lower_net[batch_index, seq_index], self.activation, self.alpha_activation)

        # Calculate sensitivities (deltas) for the upper and lower networks
        sensitivity_up = e_upper.reshape((-1, 1)) * Fprime_up
        sensitivity_low = e_lower.reshape((-1, 1)) * Fprime_low

        # Update weight gradients if trainable
        if self.train_weights:
            self.grad_w_up += np.outer(sensitivity_up.ravel(), self.input[batch_index, seq_index].ravel())
            self.grad_w_low += np.outer(sensitivity_low.ravel(), self.input[batch_index, seq_index].ravel())
            self.grad_w_up_elman += np.outer(sensitivity_up.ravel(), elman_state.ravel())
            self.grad_w_low_elman += np.outer(sensitivity_low.ravel(), elman_state.ravel())
            self.grad_w_up_jordan += np.outer(sensitivity_up.ravel(), jordan_state.ravel())
            self.grad_w_low_jordan += np.outer(sensitivity_low.ravel(), jordan_state.ravel())

        # Update bias gradients if trainable
        if self.train_bias:
            self.grad_b_up += sensitivity_up
            self.grad_b_low += sensitivity_low

        # Propagate errors to previous layer inputs and states
        error_in = np.ravel(self.upper_weight.T @ sensitivity_up + self.lower_weight.T @ sensitivity_low)
        error_elman = np.ravel(self.upper_weight_elman.T @ sensitivity_up + self.lower_weight_elman.T @ sensitivity_low)
        error_jordan = np.ravel(self.upper_weight_jordan.T @ sensitivity_up + self.lower_weight_jordan.T @ sensitivity_low)

        return error_in.reshape((-1, 1)), error_elman.reshape((-1, 1)), error_jordan.reshape((-1, 1))