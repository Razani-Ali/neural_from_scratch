import numpy as np
from activations.flexible_activation_functions import net2out, net2Fprime, net2Fstar
from initializers.weight_initializer import Dense_weight_init
from optimizers.set_optimizer import init_optimizer


class RoughActivation:
    """
    A flexible Rough Dense layer implementation that supports various activation functions, initialization options,
    and batch processing.

    Parameters:
    -----------
    input_size : int
        Number of input features.
    output_size : int
        Number of neurons in the layer.
    use_bias : bool
        Whether to include a bias term (default is True).
    batch_size : int
        Number of samples per batch (default is 32).
    activation : str
        Activation function to use. Options: 'leaky_relu', 'selu', 'elu', 'sigmoid', 'tanh'.
    alpha_upper : float, optional
        Parameter for activation functions that require an alpha value (upper network).
    alpha_lower : float, optional
        Parameter for activation functions that require an alpha value (lower network).
    lambda_upper : float, optional
        Scaling factor for 'selu' and 'elu' activations (upper network).
    lambda_lower : float, optional
        Scaling factor for 'selu' and 'elu' activations (lower network).
    train_blending : bool
        Whether to train the blending factor between upper and lower networks.
    train_weights : bool
        Whether to train weights or not.
    train_bias : bool
        Whether to train biases or not.
    train_alpha : bool
        Whether to train alpha parameters or not.
    train_lambda : bool
        Whether to train lambda parameters or not.
    weights_init_method : str
        Weight initialization method (e.g., 'he', 'xavier').
    L2_coe : float, optional
        L2 regularization coefficient.
    L1_coe : float, optional
        L1 regularization coefficient.
    weight_distribution : str
        Distribution of weights ('normal' or 'uniform').
    orthogonal_scale_factor : float
        Scale factor for orthogonal initialization.
    weights_uniform_range : tuple, optional
        Range for uniform weight distribution.

    Attributes:
    -----------
    weight : np.ndarray
        Weight matrix of shape (output_size, input_size).
    bias : np.ndarray
        Bias vector of shape (output_size, 1), initialized to zero if `use_bias` is True.
    alpha_upper : np.ndarray
        Alpha parameter for activation, shaped (output_size, 1) (upper network).
    alpha_lower : np.ndarray
        Alpha parameter for activation, shaped (output_size, 1) (lower network).
    lambda_upper : np.ndarray
        Lambda parameter for 'selu' or 'elu' activation, if applicable (upper network).
    lambda_lower : np.ndarray
        Lambda parameter for 'selu' or 'elu' activation, if applicable (lower network).
    blending_factor : np.ndarray
        Blending factor between upper and lower networks.
    """

    def __init__(self, input_size: int, output_size: int, use_bias: bool = True, batch_size: int = 32,
                 activation: str = 'leaky_relu', alpha_upper: float = None, alpha_lower: float = None,
                 lambda_upper: float = None, lambda_lower: float = None, train_blending: bool = False,
                 train_weights: bool = True, train_bias: bool = True, train_alpha: bool = True,
                 train_lambda: bool = True, weights_init_method: str = 'he', L2_coe: float = 0.0,
                 L1_coe: float = 0.0, weight_distribution: str = 'normal', 
                 orthogonal_scale_factor: float = 1.0, weights_uniform_range: tuple = None):
        """
        Initializes the RoughActivation layer with specified configurations and parameters.
        """
        self.output_size = output_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.L2_coe = L2_coe
        self.L1_coe = L1_coe
        self.use_bias = use_bias
        self.activation = activation
        self.train_weights = train_weights
        self.train_alpha = train_alpha
        self.train_bias = False if not use_bias else train_bias
        self.train_lambda = False if activation not in ['selu', 'elu', 'sin', 'cos', 'sin+cos'] else train_lambda
        self.train_blending = train_blending

        # Initialize weights
        self.weight = Dense_weight_init(input_size, output_size, method=weights_init_method, 
                                        distribution=weight_distribution, scale_factor=orthogonal_scale_factor, 
                                        ranges=weights_uniform_range)
        
        # Initialize biases
        self.bias = np.zeros((output_size, 1)) if use_bias else None
        
        # Initialize alpha parameters
        self.alpha_upper = (alpha_upper if alpha_upper is not None else (0.01 if activation == 'leaky_relu' else 1.0)) + \
                           np.zeros((output_size, 1))
        self.alpha_lower = (alpha_lower if alpha_lower is not None else (0.01 / 1.2 if activation == 'leaky_relu' else 1.0 / 1.2)) + \
                           np.zeros((output_size, 1))
        
        # Initialize lambda parameters for 'selu' or 'elu'
        self.lambda_upper = (lambda_upper if lambda_upper is not None else 1.0/1.2) + np.zeros((output_size, 1)) if \
                            activation in ['selu', 'elu', 'sin', 'cos', 'sin+cos'] else None
        self.lambda_lower = (lambda_lower if lambda_lower is not None else 1.0/1.2) + np.zeros((output_size, 1)) if \
                            activation in ['selu', 'elu', 'sin', 'cos', 'sin+cos'] else None

        # Initialize blending factor
        self.blending_factor = np.full((output_size, 1), 0.5)

        # Storage for outputs and intermediate computations
        self.net = np.zeros((batch_size, output_size, 1))
        self.final_output = np.zeros((batch_size, output_size, 1))
        self.upper_output = np.zeros((batch_size, output_size, 1))
        self.lower_output = np.zeros((batch_size, output_size, 1))
        self.minmax_reverse_stat = np.zeros((batch_size, output_size, 1))

    #################################################################

    def trainable_params(self) -> int:
        """
        Returns the total number of trainable parameters in the layer.

        Returns:
        --------
        int:
            Number of trainable parameters.
        """
        params = 0
        if self.train_weights:
            params += np.size(self.weight)
        if self.train_bias:
            params += np.size(self.bias)
        if self.train_alpha:
            params += 2 * np.size(self.alpha_upper)
        if self.train_lambda:
            params += 2 * np.size(self.lambda_upper)
        if self.train_blending:
            params += np.size(self.blending_factor)
        return params

    #################################################################

    def all_params(self) -> int:
        """
        Returns the total number of parameters in the layer.

        Returns:
        --------
        int:
            Number of parameters in the layer.
        """
        params = np.size(self.alpha_upper) * 2 + np.size(self.weight) + np.size(self.blending_factor)
        if self.use_bias:
            params += np.size(self.bias)
        if self.lambda_param is not None:
            params += np.size(self.lambda_upper) * 2
        return params

    #################################################################

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Applies the Dense layer to the input, performs linear transformations, 
        applies activation functions, and blends upper and lower outputs.

        Parameters:
        -----------
        input : np.ndarray
            Input matrix of shape (batch_size, input_size).

        Returns:
        --------
        np.ndarray:
            Activated output of the layer, shape (batch_size, output_size).
        """
        # Ensure input is reshaped to the correct format (batch_size, input_size)
        input = input.reshape((-1, self.input_size))
        self.input = input

        # Validate batch size to ensure compatibility with the layer
        if self.batch_size < input.shape[0]:
            raise ValueError('Data batch size cannot be larger than model batch size')

        # Iterate through each sample in the batch
        for batch_index, input_vector in enumerate(input):
            # Perform the linear transformation: net = weight * input + bias (if enabled)
            self.net[batch_index] = self.weight @ input_vector.reshape((-1, 1))
            if self.use_bias:
                self.net[batch_index] += self.bias

            # Apply activation functions to the transformed input
            # Apply upper and lower activation with respective alpha and lambda parameters
            up_out = net2out(
                self.net[batch_index],
                self.activation,
                alpha=self.alpha_upper,
                lambda_param=self.lambda_upper,
            )
            low_out = net2out(
                self.net[batch_index],
                self.activation,
                alpha=self.alpha_lower,
                lambda_param=self.lambda_lower,
            )

            # Combine upper and lower outputs to compute min-max reverse statistics
            concat_out = np.concatenate((up_out, low_out), axis=1)
            self.minmax_reverse_stat[batch_index] = np.argmax(concat_out).reshape((-1,1))

            # Extract max for upper and min for lower from combined outputs
            self.upper_output[batch_index] = np.max(concat_out, axis=1).reshape((-1, 1))
            self.lower_output[batch_index] = np.min(concat_out, axis=1).reshape((-1, 1))

            # Compute the final output as a weighted blend of upper and lower outputs
            self.final_output[batch_index] = (
                self.blending_factor * self.upper_output[batch_index] +
                (1 - self.blending_factor) * self.lower_output[batch_index]
            )

        # Return the final output for all input samples
        batch_index += 1
        return self.final_output[:batch_index, :, 0]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for the layer using the specified method.

        Parameters:
        -----------
        optimizer : str, optional
            The optimization algorithm to use (default is 'Adam').
            Supported optimizers include 'Adam', 'SGD', etc.
        **kwargs : dict, optional
            Additional parameters to configure the optimizer, such as learning rate.

        Returns:
        --------
        None
        """
        # Calculate the total number of trainable parameters
        num_trainable_params = self.trainable_params()

        # Initialize the optimizer with the calculated parameters and the given method
        self.Optimizer = init_optimizer(num_trainable_params, method=optimizer, **kwargs)

    #################################################################

    def update(self, grads: np.ndarray, learning_rate: float = 1e-3) -> None:
        """
        Updates the layer's trainable parameters (weights, biases, alpha, lambda) based on gradients.

        Parameters:
        -----------
        grads : np.ndarray
            Gradients of the loss with respect to each trainable parameter.
        learning_rate : float, optional
            Step size for the optimizer (default is 1e-3).

        Returns:
        --------
        None
        """
        # Compute parameter update deltas from optimizer based on gradients and learning rate
        deltas = self.Optimizer(grads, learning_rate)

        # Initialize parameter update index
        ind2 = 0

        # Update weights if trainable
        if self.train_weights:
            ind1 = ind2  # Start index for weights
            ind2 += int(np.size(self.weight))  # End index for weights
            delta_w = deltas[ind1:ind2].reshape(self.weight.shape)
            self.weight -= delta_w  # Update weights

        # Update bias if trainable
        if self.train_bias:
            ind1 = ind2  # Start index for biases
            ind2 += np.size(self.bias)  # End index for biases
            delta_bias = deltas[ind1:ind2].reshape(self.bias.shape)
            self.bias -= delta_bias  # Update bias

        # Update alpha parameter if trainable
        if self.train_alpha:
            ind1 = ind2  # Start index for alpha
            ind2 += np.size(self.alpha_upper)  # End index for alpha
            delta_alpha = deltas[ind1:ind2].reshape(self.alpha_upper.shape)
            self.alpha_upper -= delta_alpha  # Update alpha

            ind1 = ind2  # Start index for alpha
            ind2 += np.size(self.alpha_lower)  # End index for alpha
            delta_alpha = deltas[ind1:ind2].reshape(self.alpha_lower.shape)
            self.alpha_lower -= delta_alpha  # Update alpha

        # Update lambda parameter if trainable
        if self.train_lambda:
            ind1 = ind2  # Start index for lambda
            ind2 += np.size(self.lambda_lower)  # End index for lambda
            delta_lambda = deltas[ind1:ind2].reshape(self.lambda_lower.shape)
            self.lambda_lower -= delta_lambda  # Update lambda

            ind1 = ind2  # Start index for lambda
            ind2 += np.size(self.lambda_upper)  # End index for lambda
            delta_lambda = deltas[ind1:ind2].reshape(self.lambda_upper.shape)
            self.lambda_upper -= delta_lambda  # Update lambda

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
        Calculates gradients and updates the parameters during backpropagation.

        Parameters:
        -----------
        error_batch : np.ndarray
            Array of errors from the next layer, shape (batch_size, output_size).
        learning_rate : float, optional
            Learning rate for parameter updates (default is 1e-3).
        return_error : bool, optional
            If True, returns error gradients with respect to inputs (default is False).
        return_grads : bool, optional
            If True, returns computed gradients of parameters (default is False).
        modify : bool, optional
            If True, updates parameters using calculated gradients (default is True).

        Returns:
        --------
        dict or np.ndarray or None
            - Returns a dictionary with `error_in` and `gradients` if both `return_error` and `return_grads` are True.
            - Returns `error_in` if `return_error` is True and `return_grads` is False.
            - Returns `gradients` if `return_grads` is True and `return_error` is False.
            - Returns None if neither is requested.
        """
        # Initialize error gradient for inputs if return_error is True
        if return_error:
            error_in = np.zeros(self.input.shape)

        # Initialize gradients for weights, biases, alpha, and lambda parameters
        grad_w = np.zeros(self.weight.shape) if self.train_weights else None
        grad_bias = np.zeros(self.bias.shape) if self.train_bias else None
        grad_alpha_upper = np.zeros(self.alpha_upper.shape) if self.train_alpha else None
        grad_alpha_lower = np.zeros(self.alpha_lower.shape) if self.train_alpha else None
        grad_lambda_upper = np.zeros(self.lambda_upper.shape) if self.train_lambda else None
        grad_lambda_lower = np.zeros(self.lambda_lower.shape) if self.train_lambda else None
        grad_blend = np.zeros(self.blending_factor.shape) if self.train_blending else None

        # Process each error in the batch
        for batch_index, one_batch_error in enumerate(error_batch):
            one_batch_error = one_batch_error.reshape((-1, 1))
            # Calculate gradient of blending factor (alpha) if trainable
            if self.train_blending:
                grad_blend += one_batch_error * \
                            (self.upper_output[batch_index] - self.lower_output[batch_index])

            # Compute propagated error for upper and lower networks
            e_max = self.blending_factor * one_batch_error
            e_min = (1 - self.blending_factor) * one_batch_error

            # Error allocation based on network outputs and blending factor
            e_upper = e_max * np.logical_not(self.minmax_reverse_stat[batch_index]) + \
                    e_min * self.minmax_reverse_stat[batch_index]
            e_lower = e_min * np.logical_not(self.minmax_reverse_stat[batch_index]) + \
                    e_max * self.minmax_reverse_stat[batch_index]

            # Calculate derivatives for alpha and lambda if required
            if self.train_alpha or self.train_lambda:
                Fstar_upper = net2Fstar(self.net[batch_index], self.activation, self.alpha_upper, lambda_param=self.lambda_upper)
                Fstar_lower = net2Fstar(self.net[batch_index], self.activation, self.alpha_lower, lambda_param=self.lambda_lower)
                # Update alpha and lambda gradients based on activation function
                if isinstance(Fstar_upper, tuple):
                    if self.train_alpha:
                        grad_alpha_upper += e_upper * Fstar_upper[0]
                        grad_alpha_lower += e_lower * Fstar_lower[0]
                    if self.train_lambda:
                        grad_alpha_upper += e_upper * Fstar_upper[1]
                        grad_alpha_lower += e_lower * Fstar_lower[1]
                else:
                    if self.train_alpha:
                        grad_alpha_upper += e_upper * Fstar_upper
                        grad_alpha_lower += e_lower * Fstar_lower

            # Compute derivative of the activation function
            Fprime_upper = net2Fprime(self.net[batch_index], self.activation, self.alpha_upper, lambda_param=self.lambda_upper)
            Fprime_lower = net2Fprime(self.net[batch_index], self.activation, self.alpha_lower, lambda_param=self.lambda_lower)

            # Calculate sensitivity as the product of the error and activation derivative
            sensitivity = e_upper * Fprime_upper + e_lower * Fprime_lower
            # Accumulate weight gradient if trainable
            if self.train_weights:
                grad_w += np.outer(sensitivity.ravel(), self.input[batch_index].ravel())

            # Accumulate bias gradient if trainable
            if self.train_bias:
                grad_bias += sensitivity

            # Compute error gradient w.r.t input if return_error is True
            if return_error:
                error_in[batch_index] = np.ravel(self.weight.T @ sensitivity)

        # Average gradients over batch size
        if self.train_weights:
            grad_w /= error_batch.shape[0]
            grad_w += self.L1_coe * np.sign(self.weight) + self.L2_coe * self.weight
        if self.train_alpha:
            grad_alpha_upper /= error_batch.shape[0]
            grad_alpha_lower /= error_batch.shape[0]
        if self.train_bias:
            grad_bias /= error_batch.shape[0]
        if self.train_lambda:
            grad_lambda_upper /= error_batch.shape[0]
            grad_lambda_lower /= error_batch.shape[0]
        if self.train_blending:
            grad_blend /= error_batch.shape[0]

        # Collect gradients into a single array if not None
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1,1))
        if grads is not None:
            if grad_w is not None:
                grads = np.concatenate((grads, grad_w.reshape((-1,1))))
            if grad_bias is not None:
                grads = np.concatenate((grads, grad_bias.reshape((-1,1))))
            if grad_alpha_upper is not None:
                grads = np.concatenate((grads, grad_alpha_upper.reshape((-1,1))))
                grads = np.concatenate((grads, grad_alpha_lower.reshape((-1,1))))
            if grad_lambda_upper is not None:
                grads = np.concatenate((grads, grad_lambda_upper.reshape((-1,1))))
                grads = np.concatenate((grads, grad_lambda_lower.reshape((-1,1))))
            if self.train_blending:
                grads = np.concatenate((grads, grad_blend.reshape((-1,1))))

        # Update parameters if modify is True
        if modify:
            self.update(grads, learning_rate=learning_rate)

        # Return error gradients or parameter gradients based on flags
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

class TimeRoughActivation:
    """
    A flexible Rough Dense layer implementation that supports various activation functions, initialization options,
    and batch processing.

    Parameters:
    - time_steps (int): Number of time steps or sequences.
    - input_size (int): Number of input features.
    - output_size (int): Number of neurons in the layer.
    - use_bias (bool): Whether to include a bias term. Default is True.
    - batch_size (int): Number of samples per batch. Default is 32.
    - activation (str): Activation function to use. Options are 'leaky_relu', 'selu', 'elu', 'sigmoid', 'tanh'.
    - alpha_upper (float, optional): Upper bound for the alpha parameter.
    - alpha_lower (float, optional): Lower bound for the alpha parameter.
    - lambda_upper (float, optional): Upper bound for the lambda parameter.
    - lambda_lower (float, optional): Lower bound for the lambda parameter.
    - train_blending (bool): Whether to train blending factors. Default is False.
    - train_weights (bool): Whether to train weights. Default is True.
    - train_bias (bool): Whether to train bias. Default is True unless use_bias is False.
    - train_alpha (bool): Whether to train alpha parameters. Default is True.
    - train_lambda (bool): Whether to train lambda parameters. Default is True for 'selu' or 'elu'.
    - weights_init_method (str): Weight initialization method (e.g., 'he', 'xavier').
    - L2_coe (float, optional): L2 regularization coefficient. Default is 0.0.
    - L1_coe (float, optional): L1 regularization coefficient. Default is 0.0.
    - weight_distribution (str): Distribution of weights ('normal' or 'uniform'). Default is 'normal'.
    - orthogonal_scale_factor (float): Scale factor for orthogonal initialization. Default is 1.0.
    - weights_uniform_range (tuple, optional): Range for uniform weight distribution. Default is None.

    Attributes:
    - weight (np.ndarray): Weight matrix of shape (output_size, input_size).
    - bias (np.ndarray): Bias vector of shape (output_size, 1) if use_bias is True.
    - alpha_upper (np.ndarray): Upper bound for alpha parameters, shape (output_size, 1).
    - alpha_lower (np.ndarray): Lower bound for alpha parameters, shape (output_size, 1).
    - lambda_upper (np.ndarray): Upper bound for lambda parameters (for 'selu'/'elu'), shape (output_size, 1).
    - lambda_lower (np.ndarray): Lower bound for lambda parameters (for 'selu'/'elu'), shape (output_size, 1).
    - blending_factor (np.ndarray): Blending factors between upper and lower networks, shape (output_size, 1).
    - net (np.ndarray): Pre-activation values for each neuron in each batch.
    - output (np.ndarray): Activated output values for each neuron in each batch.
    - grad_* (np.ndarray): Gradient matrices for backpropagation, if respective training flags are True.
    """

    def __init__(self, 
                 time_steps: int, 
                 input_size: int, 
                 output_size: int, 
                 use_bias: bool = True, 
                 batch_size: int = 32, 
                 activation: str = 'leaky_relu',
                 alpha_upper: float = None, 
                 alpha_lower: float = None, 
                 lambda_upper: float = None, 
                 lambda_lower: float = None, 
                 train_blending: bool = False,
                 train_weights: bool = True, 
                 train_bias: bool = True, 
                 train_alpha: bool = True, 
                 train_lambda: bool = True,
                 weights_init_method: str = 'he', 
                 L2_coe: float = 0.0, 
                 L1_coe: float = 0.0,
                 weight_distribution: str = 'normal', 
                 orthogonal_scale_factor: float = 1.0, 
                 weights_uniform_range: tuple = None):
        """
        Constructor to initialize the layer's parameters and attributes.
        """
        self.output_size = output_size  # Number of neurons in the output layer.
        self.time_steps = time_steps  # Number of time steps in the sequence.
        self.input_size = input_size  # Number of input features.
        self.batch_size = batch_size  # Number of samples per batch.
        self.L2_coe = L2_coe  # L2 regularization coefficient.
        self.L1_coe = L1_coe  # L1 regularization coefficient.
        self.use_bias = use_bias  # Flag to include bias term.
        self.activation = activation  # Activation function name.
        self.train_weights = train_weights  # Flag to allow weight training.
        self.train_alpha = train_alpha  # Flag to allow alpha parameter training.
        self.train_bias = False if not use_bias else train_bias  # Bias training depends on use_bias flag.
        self.train_lambda = False if activation not in ['selu', 'elu', 'sin', 'cos', 'sin+cos'] else train_lambda  # Lambda training conditional.
        self.train_blending = train_blending  # Flag to allow blending factor training.

        # Initialize weights
        self.weight = Dense_weight_init(
            input_size=input_size, 
            output_size=output_size, 
            method=weights_init_method, 
            distribution=weight_distribution, 
            scale_factor=orthogonal_scale_factor, 
            ranges=weights_uniform_range
        )

        # Initialize bias if enabled
        if use_bias:
            self.bias = np.zeros((output_size, 1))  # Bias is initialized to zero.

        # Initialize alpha parameters
        if alpha_upper is None:
            alpha_upper = 0.01 if activation == 'leaky_relu' else 1.0
        self.alpha_upper = np.full((output_size, 1), alpha_upper)

        if alpha_lower is None:
            alpha_lower = 0.01 / 1.2 if activation == 'leaky_relu' else 1.0 / 1.2
        self.alpha_lower = np.full((output_size, 1), alpha_lower)

        # Initialize lambda parameters for SELU or ELU
        if activation in ['selu', 'elu', 'sin', 'cos', 'sin+cos']:
            self.lambda_upper = np.full((output_size, 1), lambda_upper if lambda_upper is not None else 1.0)
            self.lambda_lower = np.full((output_size, 1), lambda_lower if lambda_lower is not None else 1.0)

        # Initialize blending factor
        self.blending_factor = np.full((output_size, 1), 0.5)  # Default blending factor.

        # Initialize min-max reverse operation storage
        self.minmax_reverse_stat = np.zeros((batch_size, time_steps, output_size))

        # Initialize intermediate and final outputs
        self.net = np.zeros((batch_size, time_steps, output_size, 1))
        self.output = np.zeros((batch_size, time_steps, output_size, 1))
        self.upper_output = np.zeros((batch_size, time_steps, output_size, 1))
        self.lower_output = np.zeros((batch_size, time_steps, output_size, 1))
        self.input = np.zeros((batch_size, time_steps, input_size, 1))

        # Initialize gradients for training
        self.grad_w = np.zeros(self.weight.shape) if train_weights else None
        self.grad_bias = np.zeros(self.bias.shape) if train_bias else None
        self.grad_alpha_lower = np.zeros(self.alpha_lower.shape) if train_alpha else None
        self.grad_alpha_upper = np.zeros(self.alpha_upper.shape) if train_alpha else None
        self.grad_lambda_lower = np.zeros(self.lambda_lower.shape) if train_lambda else None
        self.grad_lambda_upper = np.zeros(self.lambda_upper.shape) if train_lambda else None
        self.grad_blend = np.zeros(self.blending_factor.shape) if train_blending else None

    #################################################################

    def trainable_params(self) -> int:
        """
        Calculate the total number of trainable parameters in the layer.

        Returns:
        --------
        int
            The total number of trainable parameters in the layer.
        """
        params = 0  # Initialize the parameter counter.
        if self.train_weights:
            # Add the size of the weight matrix to the parameter count.
            params += np.size(self.weight)
        if self.train_bias:
            # Add the size of the bias vector to the parameter count if biases are trainable.
            params += np.size(self.bias)
        if self.train_alpha:
            # Add the size of alpha parameters (upper and lower bounds).
            params += np.size(self.alpha_upper) * 2
        if self.train_lambda:
            # Add the size of lambda parameters (upper and lower bounds).
            params += np.size(self.lambda_upper) * 2
        if self.train_blending:
            # Add the size of blending factors to the parameter count.
            params += np.size(self.blending_factor)
        return params


    #################################################################

    def all_params(self) -> int:
        """
        Calculate the total number of parameters in the layer, including both trainable and non-trainable ones.

        Returns:
        --------
        int
            The total number of parameters in the layer.
        """
        # Start with the size of the weight matrix, blending factor, and alpha parameters.
        params = np.size(self.alpha_upper) * 2 + np.size(self.weight) + np.size(self.blending_factor)
        if self.use_bias:
            # Add the size of the bias vector if biases are used.
            params += np.size(self.bias)
        if self.lambda_param is not None:
            # Add the size of lambda parameters for SELU and ELU activations.
            params += np.size(self.lambda_upper) * 2
        return params


    #################################################################

    def __call__(self, batch_index: int, seq_index: int, input: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass of the model.

        Parameters:
        -----------
        batch_index : int
            The current batch index.
        seq_index : int
            The current sequence index.
        input : np.ndarray
            Input data, expected to be a vector of appropriate size.

        Returns:
        --------
        np.ndarray
            Output of the model, shaped as a vector.
        """
        # Reshape the input and store it in the corresponding position.
        self.input[batch_index, seq_index] = input.reshape((-1, 1))

        # Validate the batch size consistency.
        if self.batch_size < input.shape[0]:
            raise ValueError("Data batch size cannot be larger than the model's batch size.")

        # Apply the linear transformation: weight matrix multiplication and optional bias addition.
        self.net[batch_index, seq_index] = self.weight @ input.reshape((-1, 1))
        if self.use_bias:
            self.net[batch_index, seq_index] += self.bias

        # Calculate activation outputs for upper and lower parameters.
        up_out = net2out(self.net[batch_index, seq_index], self.activation, alpha=self.alpha_upper, lambda_param=self.lambda_upper)
        low_out = net2out(self.net[batch_index, seq_index], self.activation, alpha=self.alpha_lower, lambda_param=self.lambda_lower)

        # Combine upper and lower outputs to determine min and max values for blending.
        concat_out = np.concatenate((up_out, low_out), axis=1)
        self.minmax_reverse_stat[batch_index, seq_index] = np.argmax(concat_out, axis=1)

        # Extract the maximum (upper) and minimum (lower) outputs.
        self.upper_output[batch_index, seq_index] = np.max(concat_out, axis=1).reshape((-1, 1))
        self.lower_output[batch_index, seq_index] = np.min(concat_out, axis=1).reshape((-1, 1))

        # Compute the final output as a weighted blend of upper and lower outputs.
        self.output[batch_index, seq_index] = (
            self.blending_factor * self.upper_output[batch_index, seq_index]
            + (1 - self.blending_factor) * self.lower_output[batch_index, seq_index]
        )

        # Return the final blended output.
        return self.output[batch_index, seq_index]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for the layer using the specified method.

        Parameters:
        -----------
        optimizer : str, optional
            The optimization algorithm to use (default is 'Adam').
        **kwargs : dict, optional
            Additional parameters to configure the optimizer.

        Returns:
        --------
        None
        """
        # Initialize the optimizer using the provided method and any additional parameters.
        # The optimizer is configured based on the trainable parameters in the layer.
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)


    #################################################################

    def update(self, batch_size: int, learning_rate: float = 1e-3, grads: np.ndarray = None) -> None:
        """
        Updates the layer's weights and biases based on calculated gradients.

        Parameters:
        -----------
        batch_size : int
            The batch size used for calculating the average of gradients.
        learning_rate : float, optional
            Step size used for each iteration of parameter updates (default is 1e-3).
        grads : np.ndarray, optional
            Precomputed gradients with respect to parameters. If None, gradients are calculated internally.

        Returns:
        --------
        None
        """
        if grads is None:
            # If no gradients are provided, calculate the gradients internally.

            # Add L1 and L2 regularization gradients to weights if they are trainable.
            if self.train_weights:
                self.grad_w += self.L1_coe * np.sign(self.weight) + self.L2_coe * self.weight

            # Combine all trainable gradients into a single array.
            grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
            if grads is not None:
                if self.grad_w is not None:
                    grads = np.concatenate((grads, self.grad_w.reshape((-1, 1))))
                if self.grad_bias is not None:
                    grads = np.concatenate((grads, self.grad_bias.reshape((-1, 1))))
                if self.grad_alpha_upper is not None:
                    grads = np.concatenate((grads, self.grad_alpha_upper.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_alpha_lower.reshape((-1, 1))))
                if self.grad_lambda_upper is not None:
                    grads = np.concatenate((grads, self.grad_lambda_upper.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_lambda_lower.reshape((-1, 1))))
                if self.train_blending:
                    grads = np.concatenate((grads, self.grad_blend.reshape((-1, 1))))
                # Normalize the gradients by the batch size.
                grads /= batch_size

        # Use the optimizer to compute the parameter update deltas.
        deltas = self.Optimizer(grads, learning_rate)

        # Initialize the starting index for parameter updates.
        ind2 = 0

        # Update weights if they are trainable.
        if self.train_weights:
            ind1 = ind2  # Start index for weights.
            ind2 += np.size(self.weight)  # End index for weights.
            delta_w = deltas[ind1:ind2].reshape(self.weight.shape)
            self.weight -= delta_w  # Apply the weight update.

        # Update biases if they are trainable.
        if self.train_bias:
            ind1 = ind2  # Start index for biases.
            ind2 += np.size(self.bias)  # End index for biases.
            delta_bias = deltas[ind1:ind2].reshape(self.bias.shape)
            self.bias -= delta_bias  # Apply the bias update.

        # Update alpha parameters if they are trainable.
        if self.train_alpha:
            ind1 = ind2  # Start index for alpha_upper.
            ind2 += np.size(self.alpha_upper)  # End index for alpha_upper.
            delta_alpha = deltas[ind1:ind2].reshape(self.alpha_upper.shape)
            self.alpha_upper -= delta_alpha  # Apply the alpha_upper update.

            ind1 = ind2  # Start index for alpha_lower.
            ind2 += np.size(self.alpha_lower)  # End index for alpha_lower.
            delta_alpha = deltas[ind1:ind2].reshape(self.alpha_lower.shape)
            self.alpha_lower -= delta_alpha  # Apply the alpha_lower update.

        # Update lambda parameters if they are trainable.
        if self.train_lambda:
            ind1 = ind2  # Start index for lambda_lower.
            ind2 += np.size(self.lambda_lower)  # End index for lambda_lower.
            delta_lambda = deltas[ind1:ind2].reshape(self.lambda_lower.shape)
            self.lambda_lower -= delta_lambda  # Apply the lambda_lower update.

            ind1 = ind2  # Start index for lambda_upper.
            ind2 += np.size(self.lambda_upper)  # End index for lambda_upper.
            delta_lambda = deltas[ind1:ind2].reshape(self.lambda_upper.shape)
            self.lambda_upper -= delta_lambda  # Apply the lambda_upper update.

        # Update blending factors if they are trainable.
        if self.train_blending:
            ind1 = ind2  # Start index for blending factors.
            ind2 += np.size(self.blending_factor)  # End index for blending factors.
            delta_blend = deltas[ind1:ind2].reshape(self.blending_factor.shape)
            self.blending_factor -= delta_blend  # Apply the blending factor update.

        # Reset all gradients to zero after the update step.
        self.grad_w = self.grad_w * 0 if self.train_weights else None
        self.grad_bias = self.grad_bias * 0 if self.train_bias else None
        self.grad_alpha_upper = self.grad_alpha_upper * 0 if self.train_alpha else None
        self.grad_alpha_lower = self.grad_alpha_lower * 0 if self.train_alpha else None
        self.grad_lambda_upper = self.grad_lambda_upper * 0 if self.train_lambda else None
        self.grad_lambda_lower = self.grad_lambda_lower * 0 if self.train_lambda else None
        self.grad_blend = self.grad_blend * 0 if self.train_blending else None

    #################################################################

    def backward(self, batch_index: int, seq_index: int, error: np.ndarray) -> np.ndarray:
        """
        Computes gradients for weights, biases, and other parameters, optionally updates them, and propagates errors.

        Parameters:
        -----------
        batch_index : int
            The current batch index.
        seq_index : int
            The current sequence index.
        error : np.ndarray
            The error signal received from the subsequent layer or time step.

        Returns:
        --------
        np.ndarray
            The propagated error for the previous layer or input.
        """
        # Update blending factor gradient if it is trainable.
        if self.train_blending:
            self.grad_blend += error.reshape((-1, 1)) * (
                self.upper_output[batch_index, seq_index] - self.lower_output[batch_index, seq_index]
            )

        # Compute propagated error components for the upper and lower networks.
        e_max = self.blending_factor * error.reshape((-1, 1))
        e_min = (1 - self.blending_factor) * error.reshape((-1, 1))

        # Allocate the error based on the outputs of the upper and lower networks.
        e_upper = (
            e_max * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1)))
            + e_min * self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))
        )
        e_lower = (
            e_min * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1)))
            + e_max * self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))
        )

        # Calculate gradients for alpha and lambda parameters if they are trainable.
        if self.train_alpha or self.train_lambda:
            Fstar_upper = net2Fstar(
                self.net[batch_index, seq_index], self.activation, self.alpha_upper, lambda_param=self.lambda_upper
            )
            Fstar_lower = net2Fstar(
                self.net[batch_index, seq_index], self.activation, self.alpha_lower, lambda_param=self.lambda_lower
            )
            if isinstance(Fstar_upper, tuple):
                if self.train_alpha:
                    self.grad_alpha_upper += e_upper.reshape((-1, 1)) * Fstar_upper[0]
                    self.grad_alpha_lower += e_lower.reshape((-1, 1)) * Fstar_lower[0]
                if self.train_lambda:
                    self.grad_alpha_upper += e_upper.reshape((-1, 1)) * Fstar_upper[1]
                    self.grad_alpha_lower += e_lower.reshape((-1, 1)) * Fstar_lower[1]
            else:
                if self.train_alpha:
                    self.grad_alpha_upper += e_upper.reshape((-1, 1)) * Fstar_upper
                    self.grad_alpha_lower += e_lower.reshape((-1, 1)) * Fstar_lower

        # Compute the derivatives of the activation function for the upper and lower networks.
        Fprime_upper = net2Fprime(
            self.net[batch_index, seq_index], self.activation, self.alpha_upper, lambda_param=self.lambda_upper
        )
        Fprime_lower = net2Fprime(
            self.net[batch_index, seq_index], self.activation, self.alpha_lower, lambda_param=self.lambda_lower
        )

        # Calculate the sensitivity by combining the error and the activation function derivatives.
        sensitivity = e_upper.reshape((-1, 1)) * Fprime_upper + e_lower.reshape((-1, 1)) * Fprime_lower

        # Accumulate weight gradients if the weights are trainable.
        if self.train_weights:
            self.grad_w += np.outer(sensitivity.ravel(), self.input[batch_index, seq_index].ravel())

        # Accumulate bias gradients if biases are trainable.
        if self.train_bias:
            self.grad_bias += sensitivity

        # Propagate the error backward to the inputs of the current layer.
        error_in = np.ravel(self.weight.T @ sensitivity)

        # Return the propagated error reshaped to the appropriate form.
        return error_in.reshape((-1, 1))

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class RoughActivation1Feedback:
    """
    A flexible Rough Dense layer implementation that supports feedback mechanisms, 
    various activation functions, initialization options, and batch processing.

    Parameters:
    - time_steps (int): Number of time steps or sequences.
    - input_size (int): Number of input features.
    - output_size (int): Number of neurons in the layer.
    - feedback_size (int): The number of rows in the state vector. Defaults to `output_size` if not provided.
    - use_bias (bool): Whether to include a bias term. Default is True.
    - batch_size (int): Number of samples per batch. Default is 32.
    - activation (str): Activation function to use. Options are 'leaky_relu', 'selu', 'elu', 'sigmoid', 'tanh'.
    - alpha_upper (float, optional): Upper bound for the alpha parameter.
    - alpha_lower (float, optional): Lower bound for the alpha parameter.
    - lambda_upper (float, optional): Upper bound for the lambda parameter.
    - lambda_lower (float, optional): Lower bound for the lambda parameter.
    - train_blending (bool): Whether to train blending factors. Default is False.
    - train_weights (bool): Whether to train weights. Default is True.
    - train_bias (bool): Whether to train bias. Default is True unless `use_bias` is False.
    - train_alpha (bool): Whether to train alpha parameters. Default is True.
    - train_lambda (bool): Whether to train lambda parameters. Default is True for 'selu' or 'elu'.
    - weights_init_method (str): Weight initialization method (e.g., 'he', 'xavier').
    - L2_coe (float, optional): L2 regularization coefficient. Default is 0.0.
    - L1_coe (float, optional): L1 regularization coefficient. Default is 0.0.
    - weight_distribution (str): Distribution of weights ('normal' or 'uniform'). Default is 'normal'.
    - orthogonal_scale_factor (float): Scale factor for orthogonal initialization. Default is 1.0.
    - weights_uniform_range (tuple, optional): Range for uniform weight distribution.

    Attributes:
    - weight (np.ndarray): Weight matrix of shape (output_size, input_size).
    - weight_state (np.ndarray): Feedback weight matrix of shape (output_size, feedback_size).
    - bias (np.ndarray): Bias vector of shape (output_size, 1), initialized to zero if `use_bias` is True.
    - alpha_upper (np.ndarray): Upper bound for alpha parameters, shape (output_size, 1).
    - alpha_lower (np.ndarray): Lower bound for alpha parameters, shape (output_size, 1).
    - lambda_upper (np.ndarray): Upper bound for lambda parameters (for 'selu'/'elu'), shape (output_size, 1).
    - lambda_lower (np.ndarray): Lower bound for lambda parameters (for 'selu'/'elu'), shape (output_size, 1).
    - blending_factor (np.ndarray): Blending factors between upper and lower networks, shape (output_size, 1).
    - net (np.ndarray): Pre-activation values for each neuron in each batch.
    - output (np.ndarray): Activated output values for each neuron in each batch.
    - grad_* (np.ndarray): Gradient matrices for backpropagation, if respective training flags are True.
    """

    def __init__(self, time_steps: int, input_size: int, output_size: int, feedback_size: int = None,
                 use_bias: bool = True, batch_size: int = 32, activation: str = 'leaky_relu', 
                 alpha_upper: float = None, alpha_lower=None, lambda_upper: float = None, lambda_lower=None, 
                 train_blending: bool = False, train_weights: bool = True, train_bias: bool = True, 
                 train_alpha: bool = True, train_lambda: bool = True, weights_init_method: str = 'he', 
                 L2_coe: float = 0.0, L1_coe: float = 0.0, weight_distribution: str = 'normal',
                 orthogonal_scale_factor: float = 1.0, weights_uniform_range: tuple = None):
        """
        Initializes the RoughActivation1Feedback layer.
        """
        # Layer configuration and dimensions.
        self.output_size = output_size
        self.time_steps = time_steps
        self.input_size = input_size
        self.feedback_size = feedback_size if feedback_size is not None else output_size
        self.batch_size = batch_size
        self.L2_coe = L2_coe
        self.L1_coe = L1_coe
        self.use_bias = use_bias
        self.activation = activation
        self.train_weights = train_weights
        self.train_alpha = train_alpha
        self.train_bias = False if not use_bias else train_bias
        self.train_lambda = False if activation not in ['selu', 'elu', 'sin', 'cos', 'sin+cos'] else train_lambda
        self.train_blending = train_blending

        # Initialize weights for inputs and feedback states.
        self.weight = Dense_weight_init(input_size, output_size, method=weights_init_method, 
                                        distribution=weight_distribution, scale_factor=orthogonal_scale_factor, 
                                        ranges=weights_uniform_range)
        self.weight_state = Dense_weight_init(self.feedback_size, output_size, method=weights_init_method, 
                                              distribution=weight_distribution, scale_factor=orthogonal_scale_factor, 
                                              ranges=weights_uniform_range)

        # Initialize bias if enabled.
        if self.use_bias:
            self.bias = np.zeros((output_size, 1))

        # Set default alpha values if not provided.
        if alpha_upper is None:
            alpha_upper = 0.01 if self.activation == 'leaky_relu' else 1.0
        self.alpha_upper = alpha_upper + np.zeros((output_size, 1))  # Ensure correct shape.

        if alpha_lower is None:
            alpha_lower = 0.01 / 1.2 if self.activation == 'leaky_relu' else 1.0 / 1.2
        self.alpha_lower = alpha_lower + np.zeros((output_size, 1))  # Ensure correct shape.

        # Set default lambda values for SELU or ELU activation functions.
        self.lambda_upper = None
        self.lambda_lower = None
        if self.activation in ['selu', 'elu', 'sin', 'cos', 'sin+cos']:
            self.lambda_upper = (lambda_upper if lambda_upper is not None else 1.0) + np.zeros((output_size, 1))
            self.lambda_lower = (lambda_lower if lambda_lower is not None else 1.0) + np.zeros((output_size, 1))

        # Initialize blending factor.
        self.blending_factor = np.zeros((output_size, 1)) + 0.5

        # Track min-max reverse operations.
        self.minmax_reverse_stat = np.zeros((batch_size, time_steps, output_size))

        # Initialize intermediate and final outputs.
        self.net = np.zeros((batch_size, time_steps, output_size, 1))
        self.output = np.zeros((batch_size, time_steps, output_size, 1))
        self.upper_output = np.zeros((batch_size, time_steps, output_size, 1))
        self.lower_output = np.zeros((batch_size, time_steps, output_size, 1))
        self.input = np.zeros((batch_size, time_steps, input_size, 1))

        # Initialize gradients for training.
        self.grad_w = np.zeros(self.weight.shape) if self.train_weights else None
        self.grad_w_state = np.zeros(self.weight_state.shape) if self.train_weights else None
        self.grad_bias = np.zeros(self.bias.shape) if self.train_bias else None
        self.grad_alpha_lower = np.zeros(self.alpha_lower.shape) if self.train_alpha else None
        self.grad_alpha_upper = np.zeros(self.alpha_upper.shape) if self.train_alpha else None
        self.grad_lambda_lower = np.zeros(self.lambda_lower.shape) if self.train_lambda else None
        self.grad_lambda_upper = np.zeros(self.lambda_upper.shape) if self.train_lambda else None
        self.grad_blend = np.zeros(self.blending_factor.shape) if self.train_blending else None

    #################################################################

    def trainable_params(self) -> int:
        """
        Calculates the total number of trainable parameters in the layer.

        Returns:
        --------
        int
            The total number of trainable parameters.
        """
        params = 0  # Initialize the parameter counter.
        
        # Add weight and state weight sizes if weights are trainable.
        if self.train_weights:
            params += np.size(self.weight)
            params += np.size(self.weight_state)
        
        # Add bias size if biases are trainable.
        if self.train_bias:
            params += np.size(self.bias)
        
        # Add alpha parameters (upper and lower bounds) if trainable.
        if self.train_alpha:
            params += np.size(self.alpha_upper) * 2
        
        # Add lambda parameters (upper and lower bounds) if trainable.
        if self.train_lambda:
            params += np.size(self.lambda_upper) * 2
        
        # Add blending factors if trainable.
        if self.train_blending:
            params += np.size(self.blending_factor)
        
        return params


    #################################################################

    def all_params(self) -> int:
        """
        Calculates the total number of parameters in the layer, including both trainable and non-trainable ones.

        Returns:
        --------
        int
            The total number of parameters in the layer.
        """
        # Sum up parameters including alpha, weights, state weights, and blending factors.
        params = (
            np.size(self.alpha_upper) * 2
            + np.size(self.weight)
            + np.size(self.blending_factor)
            + np.size(self.weight_state)
        )
        
        # Add bias size if biases are used.
        if self.use_bias:
            params += np.size(self.bias)
        
        # Add lambda parameters if the activation function requires them.
        if self.lambda_param is not None:
            params += np.size(self.lambda_upper) * 2
        
        return params


    #################################################################

    def __call__(self, batch_index: int, seq_index: int, input: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the model.

        Parameters:
        -----------
        batch_index : int
            The current batch index.
        seq_index : int
            The current sequence index.
        input : np.ndarray
            Input data, expected to be a vector.
        state : np.ndarray
            Feedback state data, expected to be a vector.

        Returns:
        --------
        np.ndarray
            The output of the model, shaped as a vector.
        """
        # Store the reshaped input data in the corresponding position.
        self.input[batch_index, seq_index] = input.reshape((-1, 1))

        # Validate the batch size consistency.
        if self.batch_size < input.shape[0]:
            raise ValueError("Data batch size cannot be larger than the model's batch size.")

        # Compute the linear transformation combining input and feedback state.
        self.net[batch_index, seq_index] = (
            self.weight @ input.reshape((-1, 1))
            + self.weight_state @ state.reshape((-1, 1))
        )
        
        # Add bias if it is enabled.
        if self.use_bias:
            self.net[batch_index, seq_index] += self.bias

        # Compute activation outputs for upper and lower parameters.
        up_out = net2out(self.net[batch_index, seq_index], self.activation, alpha=self.alpha_upper, lambda_param=self.lambda_upper)
        low_out = net2out(self.net[batch_index, seq_index], self.activation, alpha=self.alpha_lower, lambda_param=self.lambda_lower)

        # Concatenate upper and lower outputs to find min and max.
        concat_out = np.concatenate((up_out, low_out), axis=1)
        self.minmax_reverse_stat[batch_index, seq_index] = np.argmax(concat_out)

        # Extract the maximum (upper) and minimum (lower) outputs.
        self.upper_output[batch_index, seq_index] = np.max(concat_out, axis=1).reshape((-1, 1))
        self.lower_output[batch_index, seq_index] = np.min(concat_out, axis=1).reshape((-1, 1))

        # Compute the final output as a weighted blend of upper and lower outputs.
        self.output[batch_index, seq_index] = (
            self.blending_factor * self.upper_output[batch_index, seq_index]
            + (1 - self.blending_factor) * self.lower_output[batch_index, seq_index]
        )

        # Return the final blended output.
        return self.output[batch_index, seq_index]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for the layer using the specified method.

        Parameters:
        -----------
        optimizer : str, optional
            The optimization algorithm to use (default is 'Adam').
        **kwargs : dict, optional
            Additional parameters to configure the optimizer.

        Returns:
        --------
        None
        """
        # Calculate the total number of trainable parameters and initialize the optimizer with the specified method.
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)


    #################################################################

    def update(self, batch_size: int, learning_rate: float = 1e-3, grads: np.ndarray = None) -> None:
        """
        Updates the layer's weights and biases based on calculated gradients.

        Parameters:
        -----------
        batch_size : int
            The data batch size used to calculate the average of gradients.
        learning_rate : float, optional
            The step size for each iteration of parameter updates (default is 1e-3).
        grads : np.ndarray, optional
            Precomputed gradients with respect to parameters. If None, gradients are calculated internally.

        Returns:
        --------
        None
        """
        if grads is None:
            # If no gradients are provided, calculate them internally.

            # Apply L1 and L2 regularization terms to the weight gradients if weights are trainable.
            if self.train_weights:
                self.grad_w += self.L1_coe * np.sign(self.weight) + self.L2_coe * self.weight
                self.grad_w_state += self.L1_coe * np.sign(self.weight_state) + self.L2_coe * self.weight_state

            # Initialize the gradients array or reshape it to include all trainable parameters.
            grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
            
            # Concatenate gradients for each trainable parameter if they exist.
            if grads is not None:
                if self.grad_w is not None:
                    grads = np.concatenate((grads, self.grad_w.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_state.reshape((-1, 1))))
                if self.grad_bias is not None:
                    grads = np.concatenate((grads, self.grad_bias.reshape((-1, 1))))
                if self.grad_alpha_upper is not None:
                    grads = np.concatenate((grads, self.grad_alpha_upper.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_alpha_lower.reshape((-1, 1))))
                if self.grad_lambda_upper is not None:
                    grads = np.concatenate((grads, self.grad_lambda_upper.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_lambda_lower.reshape((-1, 1))))
                if self.train_blending:
                    grads = np.concatenate((grads, self.grad_blend.reshape((-1, 1))))
                # Normalize gradients by dividing by the batch size.
                grads /= batch_size

        # Use the optimizer to compute parameter update deltas based on the gradients and learning rate.
        deltas = self.Optimizer(grads, learning_rate)

        # Initialize the index for iterating over deltas.
        ind2 = 0

        # Update the input weights if trainable.
        if self.train_weights:
            ind1 = ind2  # Start index for input weights.
            ind2 += int(np.size(self.weight))  # End index for input weights.
            delta_w = deltas[ind1:ind2].reshape(self.weight.shape)
            self.weight -= delta_w  # Apply the weight updates.

            ind1 = ind2  # Start index for state weights.
            ind2 += int(np.size(self.weight_state))  # End index for state weights.
            delta_w = deltas[ind1:ind2].reshape(self.weight_state.shape)
            self.weight_state -= delta_w  # Apply the state weight updates.

        # Update the biases if trainable.
        if self.train_bias:
            ind1 = ind2  # Start index for biases.
            ind2 += np.size(self.bias)  # End index for biases.
            delta_bias = deltas[ind1:ind2].reshape(self.bias.shape)
            self.bias -= delta_bias  # Apply the bias updates.

        # Update the alpha parameters if trainable.
        if self.train_alpha:
            ind1 = ind2  # Start index for upper alpha.
            ind2 += np.size(self.alpha_upper)  # End index for upper alpha.
            delta_alpha = deltas[ind1:ind2].reshape(self.alpha_upper.shape)
            self.alpha_upper -= delta_alpha  # Apply the upper alpha updates.

            ind1 = ind2  # Start index for lower alpha.
            ind2 += np.size(self.alpha_lower)  # End index for lower alpha.
            delta_alpha = deltas[ind1:ind2].reshape(self.alpha_lower.shape)
            self.alpha_lower -= delta_alpha  # Apply the lower alpha updates.

        # Update the lambda parameters if trainable.
        if self.train_lambda:
            ind1 = ind2  # Start index for lower lambda.
            ind2 += np.size(self.lambda_lower)  # End index for lower lambda.
            delta_lambda = deltas[ind1:ind2].reshape(self.lambda_lower.shape)
            self.lambda_lower -= delta_lambda  # Apply the lower lambda updates.

            ind1 = ind2  # Start index for upper lambda.
            ind2 += np.size(self.lambda_upper)  # End index for upper lambda.
            delta_lambda = deltas[ind1:ind2].reshape(self.lambda_upper.shape)
            self.lambda_upper -= delta_lambda  # Apply the upper lambda updates.

        # Update the blending factors if trainable.
        if self.train_blending:
            ind1 = ind2  # Start index for blending factors.
            ind2 += np.size(self.blending_factor)  # End index for blending factors.
            delta_blend = deltas[ind1:ind2].reshape(self.blending_factor.shape)
            self.blending_factor -= delta_blend  # Apply the blending factor updates.

        # Reset all gradients to zero after the update step.
        self.grad_w = self.grad_w * 0 if self.train_weights else None
        self.grad_w_state = self.grad_w_state * 0 if self.train_weights else None
        self.grad_bias = self.grad_bias * 0 if self.train_bias else None
        self.grad_alpha_upper = self.grad_alpha_upper * 0 if self.train_alpha else None
        self.grad_alpha_lower = self.grad_alpha_lower * 0 if self.train_alpha else None
        self.grad_lambda_upper = self.grad_alpha_upper * 0 if self.train_lambda else None
        self.grad_lambda_lower = self.grad_lambda_lower * 0 if self.train_lambda else None
        self.grad_blend = self.grad_blend * 0 if self.train_blending else None

    #################################################################

    def return_grads(self) -> np.ndarray:
        """
        Collects and returns all gradients in a single array.

        Returns:
        --------
        np.ndarray
            Concatenated gradients of all trainable parameters.
        """
        # Initialize an empty gradient array or None if there are no trainable parameters.
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))

        if grads is not None:
            # Add weight gradients for input weights with regularization if trainable.
            if self.grad_w is not None:
                grad_w = self.grad_w + (self.L1_coe * np.sign(self.weight) + self.L2_coe * self.weight)
                # Add weight gradients for state weights with regularization if trainable.
                grad_w_state = self.grad_w_state + (self.L1_coe * np.sign(self.weight_state) + self.L2_coe * self.weight_state)
                grads = np.concatenate((grads, grad_w.reshape((-1, 1))))
                grads = np.concatenate((grads, grad_w_state.reshape((-1, 1))))

            # Add bias gradients if biases are trainable.
            if self.grad_bias is not None:
                grads = np.concatenate((grads, self.grad_bias.reshape((-1, 1))))

            # Add alpha parameter gradients (upper and lower bounds) if trainable.
            if self.grad_alpha_upper is not None:
                grads = np.concatenate((grads, self.grad_alpha_upper.reshape((-1, 1))))
                grads = np.concatenate((grads, self.grad_alpha_lower.reshape((-1, 1))))

            # Add lambda parameter gradients (upper and lower bounds) if trainable.
            if self.grad_lambda_upper is not None:
                grads = np.concatenate((grads, self.grad_lambda_upper.reshape((-1, 1))))
                grads = np.concatenate((grads, self.grad_lambda_lower.reshape((-1, 1))))

            # Add blending factor gradients if trainable.
            if self.train_blending:
                grads = np.concatenate((grads, self.grad_blend.reshape((-1, 1))))

        return grads


#################################################################

    def backward(self, batch_index: int, seq_index: int, error: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes gradients for weights, biases, and optionally updates parameters. Propagates errors back 
        to inputs and state.

        Parameters:
        -----------
        batch_index : int
            The current batch index.
        seq_index : int
            The current sequence index.
        error : np.ndarray
            Error from the subsequent layer or time step.
        state : np.ndarray
            Feedback state vector needed for weight gradient calculation.

        Returns:
        --------
        tuple[np.ndarray, np.ndarray]
            - `error_in`: Propagated error for the inputs.
            - `error_state`: Propagated error for the feedback state.
        """
        # Update blending factor gradients if trainable.
        if self.train_blending:
            self.grad_blend += error.reshape((-1, 1)) * (
                self.upper_output[batch_index, seq_index] - self.lower_output[batch_index, seq_index]
            )

        # Compute propagated error components for the upper and lower networks.
        e_max = self.blending_factor * error.reshape((-1, 1))
        e_min = (1 - self.blending_factor) * error.reshape((-1, 1))

        # Allocate the error based on the outputs of the upper and lower networks.
        e_upper = (
            e_max * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1)))
            + e_min * self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))
        )
        e_lower = (
            e_min * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1)))
            + e_max * self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))
        )

        # Calculate gradients for alpha and lambda parameters if trainable.
        if self.train_alpha or self.train_lambda:
            Fstar_upper = net2Fstar(
                self.net[batch_index, seq_index], self.activation, self.alpha_upper, lambda_param=self.lambda_upper
            )
            Fstar_lower = net2Fstar(
                self.net[batch_index, seq_index], self.activation, self.alpha_lower, lambda_param=self.lambda_lower
            )
            if isinstance(Fstar_upper, tuple):
                if self.train_alpha:
                    self.grad_alpha_upper += e_upper.reshape((-1, 1)) * Fstar_upper[0]
                    self.grad_alpha_lower += e_lower.reshape((-1, 1)) * Fstar_lower[0]
                if self.train_lambda:
                    self.grad_alpha_upper += e_upper.reshape((-1, 1)) * Fstar_upper[1]
                    self.grad_alpha_lower += e_lower.reshape((-1, 1)) * Fstar_lower[1]
            else:
                if self.train_alpha:
                    self.grad_alpha_upper += e_upper.reshape((-1, 1)) * Fstar_upper
                    self.grad_alpha_lower += e_lower.reshape((-1, 1)) * Fstar_lower

        # Compute the derivatives of the activation function for the upper and lower networks.
        Fprime_upper = net2Fprime(
            self.net[batch_index, seq_index], self.activation, self.alpha_upper, lambda_param=self.lambda_upper
        )
        Fprime_lower = net2Fprime(
            self.net[batch_index, seq_index], self.activation, self.alpha_lower, lambda_param=self.lambda_lower
        )

        # Calculate the sensitivity by combining the error and the activation function derivatives.
        sensitivity = e_upper.reshape((-1, 1)) * Fprime_upper + e_lower.reshape((-1, 1)) * Fprime_lower

        # Accumulate weight gradients for input and state if trainable.
        if self.train_weights:
            self.grad_w += np.outer(sensitivity.ravel(), self.input[batch_index, seq_index].ravel())
            self.grad_w_state += np.outer(sensitivity.ravel(), state.ravel())

        # Accumulate bias gradients if trainable.
        if self.train_bias:
            self.grad_bias += sensitivity

        # Propagate the error backward to the inputs and feedback state.
        error_in = np.ravel(self.weight.T @ sensitivity)  # Error for the inputs.
        error_state = np.ravel(self.weight_state.T @ sensitivity)  # Error for the feedback state.

        # Return the propagated errors as a tuple.
        return error_in.reshape((-1, 1)), error_state.reshape((-1, 1))
    
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class RoughActivation2Feedback:
    """
    A flexible Rough Dense layer implementation that supports Elman and Jordan feedback mechanisms,
    various activation functions, initialization options, and batch processing.

    Parameters:
    -----------
    - time_steps (int): Number of time steps or sequences.
    - input_size (int): Number of input features.
    - output_size (int): Number of neurons in the layer.
    - feedback_size_jordan (int): The number of rows in the Jordan state vector.
    - use_bias (bool): Whether to include a bias term. Default is True.
    - batch_size (int): Number of samples per batch. Default is 32.
    - activation (str): Activation function to use. Options are 'leaky_relu', 'selu', 'elu', 'sigmoid', 'tanh'.
    - alpha_upper (float, optional): Upper bound for the alpha parameter.
    - alpha_lower (float, optional): Lower bound for the alpha parameter.
    - lambda_upper (float, optional): Upper bound for the lambda parameter.
    - lambda_lower (float, optional): Lower bound for the lambda parameter.
    - train_blending (bool): Whether to train blending factors. Default is False.
    - train_weights (bool): Whether to train weights. Default is True.
    - train_bias (bool): Whether to train bias. Default is True unless `use_bias` is False.
    - train_alpha (bool): Whether to train alpha parameters. Default is True.
    - train_lambda (bool): Whether to train lambda parameters. Default is True for 'selu' or 'elu'.
    - weights_init_method (str): Weight initialization method (e.g., 'he', 'xavier').
    - L2_coe (float, optional): L2 regularization coefficient. Default is 0.0.
    - L1_coe (float, optional): L1 regularization coefficient. Default is 0.0.
    - weight_distribution (str): Distribution of weights ('normal' or 'uniform'). Default is 'normal'.
    - orthogonal_scale_factor (float): Scale factor for orthogonal initialization. Default is 1.0.
    - weights_uniform_range (tuple, optional): Range for uniform weight distribution.

    Attributes:
    -----------
    - weight (np.ndarray): Weight matrix of shape (output_size, input_size).
    - weight_elman (np.ndarray): Feedback weight matrix for Elman mechanism of shape (output_size, output_size).
    - weight_jordan (np.ndarray): Feedback weight matrix for Jordan mechanism of shape (output_size, feedback_size_jordan).
    - bias (np.ndarray): Bias vector of shape (output_size, 1), initialized to zero if `use_bias` is True.
    - alpha_upper (np.ndarray): Upper bound for alpha parameters, shape (output_size, 1).
    - alpha_lower (np.ndarray): Lower bound for alpha parameters, shape (output_size, 1).
    - lambda_upper (np.ndarray): Upper bound for lambda parameters (for 'selu'/'elu'), shape (output_size, 1).
    - lambda_lower (np.ndarray): Lower bound for lambda parameters (for 'selu'/'elu'), shape (output_size, 1).
    - blending_factor (np.ndarray): Blending factors between upper and lower networks, shape (output_size, 1).
    - net (np.ndarray): Pre-activation values for each neuron in each batch.
    - output (np.ndarray): Activated output values for each neuron in each batch.
    - grad_* (np.ndarray): Gradient matrices for backpropagation, if respective training flags are True.
    """

    def __init__(self,
                 time_steps: int,
                 input_size: int,
                 output_size: int,
                 feedback_size_jordan: int = None,
                 use_bias: bool = True,
                 batch_size: int = 32,
                 activation: str = 'leaky_relu',
                 alpha_upper: float = None,
                 alpha_lower: float = None,
                 lambda_upper: float = None,
                 lambda_lower: float = None,
                 train_blending: bool = False,
                 train_weights: bool = True,
                 train_bias: bool = True,
                 train_alpha: bool = True,
                 train_lambda: bool = True,
                 weights_init_method: str = 'he',
                 L2_coe: float = 0.0,
                 L1_coe: float = 0.0,
                 weight_distribution: str = 'normal',
                 orthogonal_scale_factor: float = 1.0,
                 weights_uniform_range: tuple = None) -> None:
        """
        Initializes the RoughActivation2Feedback layer with specified parameters.
        """
        # Layer configuration and dimensions.
        self.output_size = output_size
        self.time_steps = time_steps
        self.input_size = input_size
        self.batch_size = batch_size
        self.L2_coe = L2_coe  # L2 regularization coefficient.
        self.L1_coe = L1_coe  # L1 regularization coefficient.
        self.feedback_size_elman = output_size  # Number of elements in the Elman feedback vector.
        self.feedback_size_jordan = feedback_size_jordan  # Number of elements in the Jordan feedback vector.
        self.use_bias = use_bias  # Whether to use bias terms.
        self.activation = activation  # Activation function.
        self.train_weights = train_weights  # Whether to train weights.
        self.train_alpha = train_alpha  # Whether to train alpha parameters.
        self.train_bias = False if not use_bias else train_bias  # Bias training depends on use_bias flag.
        self.train_lambda = False if activation not in ['selu', 'elu', 'sin', 'cos', 'sin+cos'] else train_lambda  # Lambda training depends on activation.
        self.train_blending = train_blending  # Whether to train blending factors.

        # Initialize weights for input, Elman feedback, and Jordan feedback.
        self.weight = Dense_weight_init(input_size, output_size, method=weights_init_method,
                                        distribution=weight_distribution, scale_factor=orthogonal_scale_factor,
                                        ranges=weights_uniform_range)
        self.weight_elman = Dense_weight_init(output_size, output_size, method=weights_init_method,
                                              distribution=weight_distribution, scale_factor=orthogonal_scale_factor,
                                              ranges=weights_uniform_range)
        self.weight_jordan = Dense_weight_init(self.feedback_size_jordan, output_size, method=weights_init_method,
                                               distribution=weight_distribution, scale_factor=orthogonal_scale_factor,
                                               ranges=weights_uniform_range)

        # Initialize bias if enabled.
        if self.use_bias:
            self.bias = np.zeros((output_size, 1))

        # Set default alpha values if not provided.
        if alpha_upper is None:
            alpha_upper = 0.01 if self.activation == 'leaky_relu' else 1.0
        self.alpha_upper = alpha_upper + np.zeros((output_size, 1))  # Ensure correct shape.

        if alpha_lower is None:
            alpha_lower = 0.01 / 1.2 if self.activation == 'leaky_relu' else 1.0 / 1.2
        self.alpha_lower = alpha_lower + np.zeros((output_size, 1))  # Ensure correct shape.

        # Set default lambda values for SELU or ELU activation functions.
        self.lambda_upper = None
        self.lambda_lower = None
        if self.activation in ['selu', 'elu', 'sin', 'cos', 'sin+cos']:
            self.lambda_upper = (lambda_upper if lambda_upper is not None else 1.0) + np.zeros((output_size, 1))
            self.lambda_lower = (lambda_lower if lambda_lower is not None else 1.0) + np.zeros((output_size, 1))

        # Initialize blending factor.
        self.blending_factor = np.zeros((output_size, 1)) + 0.5

        # Track min-max reverse operations.
        self.minmax_reverse_stat = np.zeros((batch_size, time_steps, output_size))

        # Initialize intermediate and final outputs.
        self.net = np.zeros((batch_size, time_steps, output_size, 1))
        self.output = np.zeros((batch_size, time_steps, output_size, 1))
        self.upper_output = np.zeros((batch_size, time_steps, output_size, 1))
        self.lower_output = np.zeros((batch_size, time_steps, output_size, 1))
        self.input = np.zeros((batch_size, time_steps, input_size, 1))

        # Initialize gradients for training.
        self.grad_w = np.zeros(self.weight.shape) if self.train_weights else None
        self.grad_w_elman = np.zeros(self.weight_elman.shape) if self.train_weights else None
        self.grad_w_jordan = np.zeros(self.weight_jordan.shape) if self.train_weights else None
        self.grad_bias = np.zeros(self.bias.shape) if self.train_bias else None
        self.grad_alpha_lower = np.zeros(self.alpha_lower.shape) if self.train_alpha else None
        self.grad_alpha_upper = np.zeros(self.alpha_upper.shape) if self.train_alpha else None
        self.grad_lambda_lower = np.zeros(self.lambda_lower.shape) if self.train_lambda else None
        self.grad_lambda_upper = np.zeros(self.lambda_upper.shape) if self.train_lambda else None
        self.grad_blend = np.zeros(self.blending_factor.shape) if self.train_blending else None

    #################################################################

    def trainable_params(self) -> int:
        """
        Calculates the total number of trainable parameters in the layer.

        Returns:
        --------
        int
            The total number of trainable parameters.
        """
        params = 0  # Initialize parameter counter.

        # Add weight parameters (input, Elman feedback, Jordan feedback) if trainable.
        if self.train_weights:
            params += np.size(self.weight)
            params += np.size(self.weight_elman)
            params += np.size(self.weight_jordan)

        # Add bias parameters if trainable.
        if self.train_bias:
            params += np.size(self.bias)

        # Add alpha parameters (upper and lower bounds) if trainable.
        if self.train_alpha:
            params += np.size(self.alpha_upper) * 2

        # Add lambda parameters (upper and lower bounds) if trainable.
        if self.train_lambda:
            params += np.size(self.lambda_upper) * 2

        # Add blending factor parameters if trainable.
        if self.train_blending:
            params += np.size(self.blending_factor)

        return params

    #################################################################

    def all_params(self) -> int:
        """
        Calculates the total number of parameters in the layer, including both trainable and non-trainable ones.

        Returns:
        --------
        int
            The total number of parameters in the layer.
        """
        # Start with alpha parameters, blending factors, input weights, and feedback weights.
        params = (
            np.size(self.alpha_upper) * 2
            + np.size(self.weight)
            + np.size(self.blending_factor)
            + np.size(self.weight_elman)
            + np.size(self.weight_jordan)
        )

        # Add bias parameters if biases are used.
        if self.use_bias:
            params += np.size(self.bias)

        # Add lambda parameters if activation function requires them.
        if self.lambda_param is not None:
            params += np.size(self.lambda_upper) * 2

        return params

    #################################################################

    def __call__(
        self,
        batch_index: int,
        seq_index: int,
        input: np.ndarray,
        elman_state: np.ndarray,
        jordan_state: np.ndarray,
    ) -> np.ndarray:
        """
        Performs the forward pass of the model.

        Parameters:
        -----------
        batch_index : int
            The current batch index.
        seq_index : int
            The current sequence index.
        input : np.ndarray
            Input data, expected to be a vector.
        elman_state : np.ndarray
            Feedback state vector from the current layer, expected to be a vector.
        jordan_state : np.ndarray
            Feedback state vector from the previous layer, expected to be a vector.

        Returns:
        --------
        np.ndarray
            The output of the model, shaped as a vector.
        """
        # Store the reshaped input data in the corresponding position.
        self.input[batch_index, seq_index] = input.reshape((-1, 1))

        # Validate the batch size consistency.
        if self.batch_size < input.shape[0]:
            raise ValueError("Data batch size cannot be larger than the model's batch size.")

        # Perform the linear transformation combining input, Elman feedback, and Jordan feedback.
        self.net[batch_index, seq_index] = (
            self.weight @ input.reshape((-1, 1))
            + self.weight_elman @ elman_state.reshape((-1, 1))
            + self.weight_jordan @ jordan_state.reshape((-1, 1))
        )

        # Add bias if it is enabled.
        if self.use_bias:
            self.net[batch_index, seq_index] += self.bias

        # Compute activation outputs for upper and lower parameters.
        up_out = net2out(
            self.net[batch_index, seq_index],
            self.activation,
            alpha=self.alpha_upper,
            lambda_param=self.lambda_upper,
        )
        low_out = net2out(
            self.net[batch_index, seq_index],
            self.activation,
            alpha=self.alpha_lower,
            lambda_param=self.lambda_lower,
        )

        # Concatenate upper and lower outputs to determine min and max values.
        concat_out = np.concatenate((up_out, low_out), axis=1)
        self.minmax_reverse_stat[batch_index, seq_index] = np.argmax(concat_out)

        # Extract the maximum (upper) and minimum (lower) outputs.
        self.upper_output[batch_index, seq_index] = np.max(concat_out, axis=1).reshape((-1, 1))
        self.lower_output[batch_index, seq_index] = np.min(concat_out, axis=1).reshape((-1, 1))

        # Compute the final output as a weighted blend of upper and lower outputs.
        self.output[batch_index, seq_index] = (
            self.blending_factor * self.upper_output[batch_index, seq_index]
            + (1 - self.blending_factor) * self.lower_output[batch_index, seq_index]
        )

        # Return the final blended output.
        return self.output[batch_index, seq_index]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for the layer using the specified method.

        Parameters:
        -----------
        optimizer : str, optional
            The optimization algorithm to use (default is 'Adam').
        **kwargs : dict, optional
            Additional parameters to configure the optimizer.

        Returns:
        --------
        None
        """
        # Calculate the total number of trainable parameters and initialize the optimizer.
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, batch_size: int, learning_rate: float = 1e-3, grads: np.ndarray = None) -> None:
        """
        Updates the layer's weights and biases based on the calculated gradients.

        Parameters:
        -----------
        batch_size : int
            The batch size used to calculate the average of gradients.
        learning_rate : float, optional
            The step size for each iteration of parameter updates (default is 1e-3).
        grads : np.ndarray, optional
            Precomputed gradients with respect to parameters. If None, gradients are calculated internally.

        Returns:
        --------
        None
        """
        if grads is None:
            # If no precomputed gradients are provided, calculate them internally.

            # Apply L1 and L2 regularization terms to the gradients for weights if trainable.
            if self.train_weights:
                self.grad_w += self.L1_coe * np.sign(self.weight) + self.L2_coe * self.weight
                self.grad_w_elman += self.L1_coe * np.sign(self.weight_elman) + self.L2_coe * self.weight_elman
                self.grad_w_jordan += self.L1_coe * np.sign(self.weight_jordan) + self.L2_coe * self.weight_jordan

            # Initialize or reshape the gradient array to include all trainable parameters.
            grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))

            # Concatenate gradients for each trainable parameter if they exist.
            if grads is not None:
                if self.grad_w is not None:
                    grads = np.concatenate((grads, self.grad_w.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_elman.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_jordan.reshape((-1, 1))))
                if self.grad_bias is not None:
                    grads = np.concatenate((grads, self.grad_bias.reshape((-1, 1))))
                if self.grad_alpha_upper is not None:
                    grads = np.concatenate((grads, self.grad_alpha_upper.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_alpha_lower.reshape((-1, 1))))
                if self.grad_lambda_upper is not None:
                    grads = np.concatenate((grads, self.grad_lambda_upper.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_lambda_lower.reshape((-1, 1))))
                if self.train_blending:
                    grads = np.concatenate((grads, self.grad_blend.reshape((-1, 1))))
                # Normalize gradients by dividing by the batch size.
                grads /= batch_size

        # Compute the parameter update deltas from the optimizer using gradients and the learning rate.
        deltas = self.Optimizer(grads, learning_rate)

        # Initialize the index for iterating over the parameter deltas.
        ind2 = 0

        # Update the input weights if trainable.
        if self.train_weights:
            ind1 = ind2  # Start index for input weights.
            ind2 += int(np.size(self.weight))  # End index for input weights.
            delta_w = deltas[ind1:ind2].reshape(self.weight.shape)
            self.weight -= delta_w  # Apply the weight updates.

            ind1 = ind2  # Start index for Elman feedback weights.
            ind2 += int(np.size(self.weight_elman))  # End index for Elman feedback weights.
            delta_w = deltas[ind1:ind2].reshape(self.weight_elman.shape)
            self.weight_elman -= delta_w  # Apply the Elman weight updates.

            ind1 = ind2  # Start index for Jordan feedback weights.
            ind2 += int(np.size(self.weight_jordan))  # End index for Jordan feedback weights.
            delta_w = deltas[ind1:ind2].reshape(self.weight_jordan.shape)
            self.weight_jordan -= delta_w  # Apply the Jordan weight updates.

        # Update the biases if trainable.
        if self.train_bias:
            ind1 = ind2  # Start index for biases.
            ind2 += np.size(self.bias)  # End index for biases.
            delta_bias = deltas[ind1:ind2].reshape(self.bias.shape)
            self.bias -= delta_bias  # Apply the bias updates.

        # Update the alpha parameters if trainable.
        if self.train_alpha:
            ind1 = ind2  # Start index for upper alpha parameters.
            ind2 += np.size(self.alpha_upper)  # End index for upper alpha parameters.
            delta_alpha = deltas[ind1:ind2].reshape(self.alpha_upper.shape)
            self.alpha_upper -= delta_alpha  # Apply the upper alpha updates.

            ind1 = ind2  # Start index for lower alpha parameters.
            ind2 += np.size(self.alpha_lower)  # End index for lower alpha parameters.
            delta_alpha = deltas[ind1:ind2].reshape(self.alpha_lower.shape)
            self.alpha_lower -= delta_alpha  # Apply the lower alpha updates.

        # Update the lambda parameters if trainable.
        if self.train_lambda:
            ind1 = ind2  # Start index for lower lambda parameters.
            ind2 += np.size(self.lambda_lower)  # End index for lower lambda parameters.
            delta_lambda = deltas[ind1:ind2].reshape(self.lambda_lower.shape)
            self.lambda_lower -= delta_lambda  # Apply the lower lambda updates.

            ind1 = ind2  # Start index for upper lambda parameters.
            ind2 += np.size(self.lambda_upper)  # End index for upper lambda parameters.
            delta_lambda = deltas[ind1:ind2].reshape(self.lambda_upper.shape)
            self.lambda_upper -= delta_lambda  # Apply the upper lambda updates.

        # Update the blending factors if trainable.
        if self.train_blending:
            ind1 = ind2  # Start index for blending factors.
            ind2 += np.size(self.blending_factor)  # End index for blending factors.
            delta_blend = deltas[ind1:ind2].reshape(self.blending_factor.shape)
            self.blending_factor -= delta_blend  # Apply the blending factor updates.

        # Reset all gradients to zero after the update step.
        self.grad_w = self.grad_w * 0 if self.train_weights else None
        self.grad_w_elman = self.grad_w_elman * 0 if self.train_weights else None
        self.grad_w_jordan = self.grad_w_jordan * 0 if self.train_weights else None
        self.grad_bias = self.grad_bias * 0 if self.train_bias else None
        self.grad_alpha_upper = self.grad_alpha_upper * 0 if self.train_alpha else None
        self.grad_alpha_lower = self.grad_alpha_lower * 0 if self.train_alpha else None
        self.grad_lambda_upper = self.grad_alpha_upper * 0 if self.train_lambda else None
        self.grad_lambda_lower = self.grad_lambda_lower * 0 if self.train_lambda else None
        self.grad_blend = self.grad_blend * 0 if self.train_blending else None

    #################################################################

    def return_grads(self) -> np.ndarray:
        """
        Collects all the gradients for the trainable parameters into a single array.

        Returns:
        --------
        np.ndarray
            A single concatenated array containing all gradients for the trainable parameters.
            If there are no trainable parameters, returns None.
        """
        # Initialize an empty gradient array or None if there are no trainable parameters.
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))

        if grads is not None:
            # Add weight gradients for input, Elman, and Jordan weights with regularization.
            if self.grad_w is not None:
                grad_w = self.grad_w + (self.L1_coe * np.sign(self.weight) + self.L2_coe * self.weight)
                grad_w_elman = self.grad_w_elman + (self.L1_coe * np.sign(self.weight_elman) + self.L2_coe * self.weight_elman)
                grad_w_jordan = self.grad_w_jordan + (self.L1_coe * np.sign(self.weight_jordan) + self.L2_coe * self.weight_jordan)
                grads = np.concatenate((grads, grad_w.reshape((-1, 1))))
                grads = np.concatenate((grads, grad_w_elman.reshape((-1, 1))))
                grads = np.concatenate((grads, grad_w_jordan.reshape((-1, 1))))

            # Add bias gradients if biases are trainable.
            if self.grad_bias is not None:
                grads = np.concatenate((grads, self.grad_bias.reshape((-1, 1))))

            # Add alpha parameter gradients (upper and lower bounds) if trainable.
            if self.grad_alpha_upper is not None:
                grads = np.concatenate((grads, self.grad_alpha_upper.reshape((-1, 1))))
                grads = np.concatenate((grads, self.grad_alpha_lower.reshape((-1, 1))))

            # Add lambda parameter gradients (upper and lower bounds) if trainable.
            if self.grad_lambda_upper is not None:
                grads = np.concatenate((grads, self.grad_lambda_upper.reshape((-1, 1))))
                grads = np.concatenate((grads, self.grad_lambda_lower.reshape((-1, 1))))

            # Add blending factor gradients if trainable.
            if self.train_blending:
                grads = np.concatenate((grads, self.grad_blend.reshape((-1, 1))))

        return grads


    #################################################################

    def backward(self, batch_index: int, seq_index: int, error: np.ndarray, elman_state: np.ndarray, jordan_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes gradients for weights, biases, and other parameters. Propagates errors back to inputs,
        Elman feedback, and Jordan feedback.

        Parameters:
        -----------
        batch_index : int
            The current batch index.
        seq_index : int
            The current sequence index.
        error : np.ndarray
            The error signal received from the subsequent layer or time step.
        elman_state : np.ndarray
            Feedback state vector from the current layer, used for weight gradient calculation.
        jordan_state : np.ndarray
            Feedback state vector from the previous layer, used for weight gradient calculation.

        Returns:
        --------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            - `error_in`: Propagated error for the inputs.
            - `error_elman`: Propagated error for the Elman feedback.
            - `error_jordan`: Propagated error for the Jordan feedback.
        """
        # Update blending factor gradients if trainable.
        if self.train_blending:
            self.grad_blend += error.reshape((-1, 1)) * (
                self.upper_output[batch_index, seq_index] - self.lower_output[batch_index, seq_index]
            )

        # Compute propagated error components for the upper and lower networks.
        e_max = self.blending_factor * error.reshape((-1, 1))
        e_min = (1 - self.blending_factor) * error.reshape((-1, 1))

        # Allocate the error based on the outputs of the upper and lower networks.
        e_upper = (
            e_max * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1)))
            + e_min * self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))
        )
        e_lower = (
            e_min * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1)))
            + e_max * self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))
        )

        # Calculate gradients for alpha and lambda parameters if trainable.
        if self.train_alpha or self.train_lambda:
            Fstar_upper = net2Fstar(
                self.net[batch_index, seq_index], self.activation, self.alpha_upper, lambda_param=self.lambda_upper
            )
            Fstar_lower = net2Fstar(
                self.net[batch_index, seq_index], self.activation, self.alpha_lower, lambda_param=self.lambda_lower
            )
            if isinstance(Fstar_upper, tuple):
                if self.train_alpha:
                    self.grad_alpha_upper += e_upper.reshape((-1, 1)) * Fstar_upper[0]
                    self.grad_alpha_lower += e_lower.reshape((-1, 1)) * Fstar_lower[0]
                if self.train_lambda:
                    self.grad_alpha_upper += e_upper.reshape((-1, 1)) * Fstar_upper[1]
                    self.grad_alpha_lower += e_lower.reshape((-1, 1)) * Fstar_lower[1]
            else:
                if self.train_alpha:
                    self.grad_alpha_upper += e_upper.reshape((-1, 1)) * Fstar_upper
                    self.grad_alpha_lower += e_lower.reshape((-1, 1)) * Fstar_lower

        # Compute the derivatives of the activation function for the upper and lower networks.
        Fprime_upper = net2Fprime(
            self.net[batch_index, seq_index], self.activation, self.alpha_upper, lambda_param=self.lambda_upper
        )
        Fprime_lower = net2Fprime(
            self.net[batch_index, seq_index], self.activation, self.alpha_lower, lambda_param=self.lambda_lower
        )

        # Calculate the sensitivity by combining the error and the activation function derivatives.
        sensitivity = e_upper.reshape((-1, 1)) * Fprime_upper + e_lower.reshape((-1, 1)) * Fprime_lower

        # Accumulate weight gradients for input, Elman feedback, and Jordan feedback if trainable.
        if self.train_weights:
            self.grad_w += np.outer(sensitivity.ravel(), self.input[batch_index, seq_index].ravel())
            self.grad_w_elman += np.outer(sensitivity.ravel(), elman_state.ravel())
            self.grad_w_jordan += np.outer(sensitivity.ravel(), jordan_state.ravel())

        # Accumulate bias gradients if trainable.
        if self.train_bias:
            self.grad_bias += sensitivity

        # Propagate the error backward to the inputs, Elman feedback, and Jordan feedback.
        error_in = np.ravel(self.weight.T @ sensitivity)  # Error for the inputs.
        error_elman = np.ravel(self.weight_elman.T @ sensitivity)  # Error for the Elman feedback.
        error_jordan = np.ravel(self.weight_jordan.T @ sensitivity)  # Error for the Jordan feedback.

        # Return the propagated errors as a tuple.
        return error_in.reshape((-1, 1)), error_elman.reshape((-1, 1)), error_jordan.reshape((-1, 1))
