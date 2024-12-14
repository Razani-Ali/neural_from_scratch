import numpy as np
from activations.flexible_activation_functions import net2out, net2Fprime, net2Fstar
from initializers.weight_initializer import Dense_weight_init
from optimizers.set_optimizer import init_optimizer

class FlexibleRough:
    """
    Implements a neural network layer-like structure with dual upper and lower networks,
    controlled by an alpha blending factor. This layer can blend outputs from two networks
    and apply different activation functions.

    Parameters:
        input_size (int): Number of input features.
        output_size (int): Number of output neurons.
        use_bias (bool, optional): Whether to use bias in the networks. Default is True.
        batch_size (int, optional): Number of samples in each batch. Default is 32.
        train_weights (bool, optional): Whether weights are trainable. Default is True.
        train_bias (bool, optional): Whether biases are trainable. Default is True.
        train_blending (bool, optional): Whether the blending factor is trainable. Default is False.
        alpha (float, optional): Alpha parameter for activation functions. Default is None.
        lambda_ (float, optional): Lambda parameter for specific activation functions. Default is None.
        train_alpha (bool, optional): Whether alpha is trainable. Default is True.
        train_lambda (bool, optional): Whether lambda is trainable. Default is True.
        activation (str, optional): Activation function ('relu', 'sigmoid', etc.). Default is 'sigmoid'.
        weights_uniform_range (tuple, optional): Range for initializing weights. Default is (-1, 1).
        L2_coe (float, optional): L2 regularization coefficient. Default is 0.0.
        L1_coe (float, optional): L1 regularization coefficient. Default is 0.0.

    Attributes:
        upper_weight (np.ndarray): Weights for the upper network.
        lower_weight (np.ndarray): Weights for the lower network.
        blending_factor (np.ndarray): Blending factor for combining upper and lower outputs.
        upper_net (np.ndarray): Pre-activation outputs from the upper network.
        lower_net (np.ndarray): Pre-activation outputs from the lower network.
        final_output (np.ndarray): Final blended output after activation.
    """

    def __init__(self, input_size: int, output_size: int, use_bias: bool = True, batch_size: int = 32,
                 train_weights: bool = True, train_bias: bool = True, train_blending: bool = False,
                 alpha: float = None, lambda_=None, train_alpha: bool = True, train_lambda: bool = True,
                 activation: str = 'sigmoid', weights_uniform_range: tuple = (-1, 1),
                 L2_coe: float = 0.0, L1_coe: float = 0.0):
        """
        Initializes the FlexibleRough layer.

        Parameters:
            input_size (int): Number of input features.
            output_size (int): Number of output neurons.
            use_bias (bool): Whether to use bias in the layer. Default is True.
            batch_size (int): Number of samples in each batch. Default is 32.
            train_weights (bool): Whether weights are trainable. Default is True.
            train_bias (bool): Whether biases are trainable. Default is True.
            train_blending (bool): Whether the blending factor is trainable. Default is False.
            alpha (float): Alpha parameter for activation functions. Default is None.
            lambda_ (float): Lambda parameter for specific activation functions. Default is None.
            train_alpha (bool): Whether alpha is trainable. Default is True.
            train_lambda (bool): Whether lambda is trainable. Default is True.
            activation (str): Activation function. Default is 'sigmoid'.
            weights_uniform_range (tuple): Range for initializing weights. Default is (-1, 1).
            L2_coe (float): L2 regularization coefficient. Default is 0.0.
            L1_coe (float): L1 regularization coefficient. Default is 0.0.

        Returns:
            None
        """
        # Layer configurations
        self.output_size = output_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.L2_coe = L2_coe
        self.L1_coe = L1_coe
        self.use_bias = use_bias
        self.activation = activation
        self.train_weights = train_weights
        self.train_bias = train_bias if use_bias else False
        self.train_blending = train_blending
        self.train_alpha = train_alpha
        self.train_lambda = train_lambda if activation in ['selu', 'elu'] else False

        # Split weight initialization ranges into upper and lower halves
        middle = (weights_uniform_range[0] + weights_uniform_range[1]) / 2
        upper_range = (weights_uniform_range[0], middle)
        lower_range = (middle, weights_uniform_range[1])

        # Initialize weights
        self.upper_weight = Dense_weight_init(input_size, output_size, method="uniform", ranges=upper_range)
        self.lower_weight = Dense_weight_init(input_size, output_size, method="uniform", ranges=lower_range)

        # Initialize biases
        if self.use_bias:
            self.upper_bias = np.zeros((output_size, 1))
            self.lower_bias = np.zeros((output_size, 1))

        # Initialize blending factor
        self.blending_factor = np.zeros((output_size, 1)) + 0.5

        # Set default alpha and lambda values
        if alpha is None:
            alpha = 0.01 if self.activation == 'leaky_relu' else 1.0
        self.alpha = alpha + np.zeros((output_size, 1))

        self.lambda_param = None
        if self.activation in ['selu', 'elu']:
            self.lambda_param = (lambda_ if lambda_ is not None else 1.0) + np.zeros((output_size, 1))

        # Used to store min-max reverse operation results
        self.minmax_reverse_stat = np.zeros((batch_size, output_size, 1))

        # Initialize intermediate outputs
        self.upper_net = np.zeros((batch_size, output_size, 1))
        self.lower_net = np.zeros((batch_size, output_size, 1))
        self.final_output = np.zeros((batch_size, output_size, 1))
        self.upper_output = np.zeros((batch_size, output_size, 1))
        self.lower_output = np.zeros((batch_size, output_size, 1))

    #################################################################

    def trainable_params(self) -> int:
        """
        Returns the total number of trainable parameters in the layer.

        Returns:
            int: Total number of trainable parameters.
        """
        params = 0
        if self.train_weights:
            params += np.size(self.upper_weight) + np.size(self.lower_weight)  # Weights
        if self.train_bias:
            params += np.size(self.upper_bias) + np.size(self.lower_bias)  # Biases
        if self.train_blending:
            params += np.size(self.blending_factor)  # Blending factor
        if self.train_alpha:
            params += np.size(self.alpha)  # Alpha parameters
        if self.train_lambda:
            params += np.size(self.lambda_param)  # Lambda parameters
        return params

    #################################################################

    def all_params(self) -> int:
        """
        Returns the total number of parameters in the layer, including non-trainable parameters.

        Returns:
            int: Total number of parameters.
        """
        params = np.size(self.upper_weight) + np.size(self.lower_weight) + np.size(self.alpha)
        params += np.size(self.blending_factor)
        if self.use_bias:
            params += np.size(self.upper_bias) + np.size(self.lower_bias)
        if self.lambda_param is not None:
            params += np.size(self.lambda_param)
        return params

    #################################################################

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass through the layer.

        Parameters:
            input (np.ndarray): Input data of shape (batch_size, input_size).

        Returns:
            np.ndarray: Output of the layer of shape (batch_size, output_size).
        """
        input = input.reshape((-1, self.input_size))  # Ensure correct input shape
        self.input = input

        # Check batch size
        if self.batch_size < input.shape[0]:
            raise ValueError('Data batch size cannot exceed the model batch size.')

        # Loop through each sample in the batch
        for batch_index, input_vector in enumerate(input):
            # Compute upper and lower nets
            self.upper_net[batch_index] = self.upper_weight @ input_vector.reshape((-1, 1))
            self.lower_net[batch_index] = self.lower_weight @ input_vector.reshape((-1, 1))

            # Add biases if applicable
            if self.use_bias:
                self.upper_net[batch_index] += self.upper_bias
                self.lower_net[batch_index] += self.lower_bias

            # Apply activation functions
            up_out = net2out(self.upper_net[batch_index], self.activation, alpha=self.alpha, lambda_param=self.lambda_param)
            low_out = net2out(self.lower_net[batch_index], self.activation, alpha=self.alpha, lambda_param=self.lambda_param)

            # Concatenate upper and lower outputs to find min and max
            concat_out = np.concatenate((up_out, low_out), axis=1)
            self.minmax_reverse_stat[batch_index] = np.argmax(concat_out).reshape(-1,1)

            # Get max for upper and min for lower
            self.upper_output[batch_index] = np.max(concat_out, axis=1).reshape((-1, 1))
            self.lower_output[batch_index] = np.min(concat_out, axis=1).reshape((-1, 1))
            self.final_output[batch_index] = self.blending_factor * up_out + (1 - self.blending_factor) * low_out

        # Return the final output for all input samples
        batch_index += 1
        return self.final_output[:batch_index, :, 0]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for the layer.

        Parameters:
            optimizer (str): The optimizer to use (e.g., 'Adam', 'SGD'). Default is 'Adam'.
            **kwargs: Additional parameters for configuring the optimizer.

        Returns:
            None
        """
        # Initialize optimizer based on trainable parameters and given configuration
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, grads: np.ndarray, learning_rate: float = 1e-3) -> None:
        """
        Updates the trainable parameters of the layer using gradients.

        Parameters:
            grads (np.ndarray): Gradients for all trainable parameters.
            learning_rate (float): Learning rate for the updates. Default is 1e-3.

        Returns:
            None
        """
        # Compute parameter updates using optimizer
        deltas = self.Optimizer(grads, learning_rate)

        # Track position in gradient array
        ind2 = 0

        # Update weights
        if self.train_weights:
            # Upper weights
            ind1 = ind2
            ind2 += int(np.size(self.upper_weight))
            delta_w = deltas[ind1:ind2].reshape(self.upper_weight.shape)
            self.upper_weight -= delta_w

            # Lower weights
            ind1 = ind2
            ind2 += int(np.size(self.lower_weight))
            delta_w = deltas[ind1:ind2].reshape(self.lower_weight.shape)
            self.lower_weight -= delta_w

        # Update biases
        if self.train_bias:
            # Upper bias
            ind1 = ind2
            ind2 += int(np.size(self.upper_bias))
            delta_bias = deltas[ind1:ind2].reshape(self.upper_bias.shape)
            self.upper_bias -= delta_bias

            # Lower bias
            ind1 = ind2
            ind2 += int(np.size(self.lower_bias))
            delta_bias = deltas[ind1:ind2].reshape(self.lower_bias.shape)
            self.lower_bias -= delta_bias

        # Update alpha
        if self.train_alpha:
            ind1 = ind2
            ind2 += int(np.size(self.alpha))
            delta_alpha = deltas[ind1:ind2].reshape(self.alpha.shape)
            self.alpha -= delta_alpha

        # Update lambda
        if self.train_lambda:
            ind1 = ind2
            ind2 += int(np.size(self.lambda_param))
            delta_lambda = deltas[ind1:ind2].reshape(self.lambda_param.shape)
            self.lambda_param -= delta_lambda

        # Update blending factor
        if self.train_blending:
            ind1 = ind2
            ind2 += int(np.size(self.blending_factor))
            delta_blend = deltas[ind1:ind2].reshape(self.blending_factor.shape)
            self.blending_factor -= delta_blend

    #################################################################

    def backward(self, error_batch: np.ndarray, learning_rate: float = 1e-3, 
                 return_error: bool = False, return_grads: bool = False, modify: bool = True):
        """
        Backward pass for computing gradients and optionally updating parameters.

        Parameters:
            error_batch (np.ndarray): Errors propagated from the next layer (batch_size, output_size).
            learning_rate (float): Learning rate for parameter updates. Default is 1e-3.
            return_error (bool): Whether to return the input error for further backpropagation. Default is False.
            return_grads (bool): Whether to return the gradients for inspection. Default is False.
            modify (bool): Whether to apply updates to the parameters. Default is True.

        Returns:
            dict or np.ndarray or None:
                - If both `return_error` and `return_grads` are True, returns a dictionary with:
                  {'error_in': error_in, 'gradients': grads}.
                - If `return_error` is True, returns `error_in`.
                - If `return_grads` is True, returns `gradients`.
                - If neither is True, returns None.
        """
        # Initialize propagated error if required
        if return_error:
            error_in = np.zeros(self.input.shape)

        # Initialize gradients for weights, biases, alpha, lambda, and blending factor
        grad_w_up = np.zeros(self.upper_weight.shape) if self.train_weights else None
        grad_w_low = np.zeros(self.lower_weight.shape) if self.train_weights else None
        grad_bias_up = np.zeros(self.upper_bias.shape) if self.train_bias else None
        grad_bias_low = np.zeros(self.lower_bias.shape) if self.train_bias else None
        grad_alpha = np.zeros(self.alpha.shape) if self.train_alpha else None
        grad_lambda = np.zeros(self.lambda_param.shape) if self.train_lambda else None
        grad_blend = np.zeros(self.blending_factor.shape) if self.train_blending else None

        # Iterate through each error in the batch
        for batch_index, one_batch_error in enumerate(error_batch):
            one_batch_error = one_batch_error.reshape((-1, 1))
            # Compute gradient of blending factor (alpha) if trainable
            if self.train_blending:
                grad_blend += one_batch_error * \
                              (self.upper_output[batch_index] - self.lower_output[batch_index])

            # Error allocation to upper and lower networks
            e_max = self.blending_factor * one_batch_error
            e_min = (1 - self.blending_factor) * one_batch_error

            e_upper = e_max * np.logical_not(self.minmax_reverse_stat[batch_index]) + \
                      e_min * self.minmax_reverse_stat[batch_index]
            e_lower = e_min * np.logical_not(self.minmax_reverse_stat[batch_index]) + \
                      e_max * self.minmax_reverse_stat[batch_index]

            # Calculate derivatives for alpha and lambda if applicable
            if self.train_alpha or self.train_lambda:
                Fstar_upper = net2Fstar(self.upper_net[batch_index], self.activation, self.alpha, lambda_param=self.lambda_param)
                Fstar_lower = net2Fstar(self.lower_net[batch_index], self.activation, self.alpha, lambda_param=self.lambda_param)

                if self.activation in ['selu', 'elu']:
                    if self.train_alpha:
                        grad_alpha += e_upper * Fstar_upper[0]
                        grad_alpha += e_lower * Fstar_lower[0]
                    if self.train_lambda:
                        grad_lambda += e_upper * Fstar_upper[1]
                        grad_lambda += e_lower * Fstar_lower[1]
                else:
                    if self.train_alpha:
                        grad_alpha += e_upper * Fstar_upper
                        grad_alpha += e_lower * Fstar_lower

            # Compute derivative of the activation function
            Fprime_up = net2Fprime(self.upper_net[batch_index], self.activation, self.alpha, lambda_param=self.lambda_param)
            Fprime_low = net2Fprime(self.lower_net[batch_index], self.activation, self.alpha, lambda_param=self.lambda_param)

            # Calculate sensitivities
            sensitivity_up = e_upper * Fprime_up
            sensitivity_low = e_lower * Fprime_low

            # Accumulate gradients for weights
            if self.train_weights:
                grad_w_up += np.outer(sensitivity_up.ravel(), self.input[batch_index].ravel())
                grad_w_low += np.outer(sensitivity_low.ravel(), self.input[batch_index].ravel())

            # Accumulate gradients for biases
            if self.train_bias:
                grad_bias_up += sensitivity_up
                grad_bias_low += sensitivity_low

            # Propagate error to previous layer
            if return_error:
                error_in[batch_index] = np.ravel(self.upper_weight.T @ sensitivity_up + self.lower_weight.T @ sensitivity_low)

        # Average gradients over the batch
        if self.train_weights:
            grad_w_up /= error_batch.shape[0]
            grad_w_low /= error_batch.shape[0]
        if self.train_bias:
            grad_bias_up /= error_batch.shape[0]
            grad_bias_low /= error_batch.shape[0]
        if self.train_alpha:
            grad_alpha /= error_batch.shape[0]
        if self.train_lambda:
            grad_lambda /= error_batch.shape[0]
        if self.train_blending:
            grad_blend /= error_batch.shape[0]

        # Prepare gradients array for update
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
        if grads is not None:
            if grad_w_up is not None:
                grads = np.concatenate((grads, grad_w_up.reshape((-1, 1))))
                grads = np.concatenate((grads, grad_w_low.reshape((-1, 1))))
            if grad_bias_up is not None:
                grads = np.concatenate((grads, grad_bias_up.reshape((-1, 1))))
                grads = np.concatenate((grads, grad_bias_low.reshape((-1, 1))))
            if grad_alpha is not None:
                grads = np.concatenate((grads, grad_alpha.reshape((-1, 1))))
            if grad_lambda is not None:
                grads = np.concatenate((grads, grad_lambda.reshape((-1, 1))))
            if grad_blend is not None:
                grads = np.concatenate((grads, grad_blend.reshape((-1, 1))))

        # Apply updates if modify is True
        if modify and grads is not None:
            self.update(grads, learning_rate=learning_rate)

        # Return results based on flags
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

class TimeFlexibleRough:
    """
    A custom neural network layer designed for time-series data with flexible weight blending 
    between two parallel networks (upper and lower) based on an adjustable alpha parameter. 
    This layer supports advanced functionalities such as bias usage, regularization, and custom 
    activation functions.

    Attributes:
    ----------
    time_steps : int
        Number of time steps or sequences to process.
    input_size : int
        Number of input neurons/features.
    output_size : int
        Number of output neurons.
    batch_size : int
        Number of samples in each batch.
    use_bias : bool
        Indicates whether biases are used in the network computations.
    activation : str
        Specifies the activation function used in the layer (e.g., 'sigmoid', 'relu').
    train_weights : bool
        Determines if weights are trainable.
    train_bias : bool
        Determines if biases are trainable.
    train_blending : bool
        Indicates if the blending factor between upper and lower networks is trainable.
    train_alpha : bool
        Indicates if the alpha parameter (scaling for activation functions) is trainable.
    train_lambda : bool
        Indicates if the lambda parameter (used in 'selu' or 'elu') is trainable.
    L2_coe : float
        Coefficient for L2 regularization.
    L1_coe : float
        Coefficient for L1 regularization.
    upper_weight : np.ndarray
        Weights for the upper network.
    lower_weight : np.ndarray
        Weights for the lower network.
    upper_bias : Optional[np.ndarray]
        Bias for the upper network (if enabled).
    lower_bias : Optional[np.ndarray]
        Bias for the lower network (if enabled).
    blending_factor : np.ndarray
        Blending factor that determines the weightage of upper and lower networks in the output.
    alpha : np.ndarray
        Alpha parameter for activation function scaling.
    lambda_param : Optional[np.ndarray]
        Lambda parameter for specific activation functions ('selu' or 'elu').
    upper_net : np.ndarray
        Stores intermediate output from the upper network during forward passes.
    lower_net : np.ndarray
        Stores intermediate output from the lower network during forward passes.
    output : np.ndarray
        Final output of the layer after blending.
    input : np.ndarray
        Stores input data for forward and backward computations.
    grad_w_up : Optional[np.ndarray]
        Gradients for the upper network weights.
    grad_w_low : Optional[np.ndarray]
        Gradients for the lower network weights.
    grad_bias_up : Optional[np.ndarray]
        Gradients for the upper network biases.
    grad_bias_low : Optional[np.ndarray]
        Gradients for the lower network biases.
    grad_alpha : Optional[np.ndarray]
        Gradients for the alpha parameter.
    grad_lambda : Optional[np.ndarray]
        Gradients for the lambda parameter.
    grad_blend : Optional[np.ndarray]
        Gradients for the blending factor.
    """

    def __init__(self, time_steps: int, input_size: int, output_size: int, use_bias: bool = True, batch_size: int = 32,
                train_weights: bool = True, train_bias: bool = True, train_blending: bool = False,
                alpha: float = None, lambda_: float = None, train_alpha: bool = True, train_lambda: bool = True,
                activation: str = 'sigmoid', weights_uniform_range: tuple = (-1, 1),
                L2_coe: float = 0.0, L1_coe: float = 0.0) -> None:
        """
        Initializes the TimeFlexibleRough neural network layer.
        
        Parameters:
        ----------
        time_steps : int
            Number of time steps or sequences to process.
        input_size : int
            Number of input neurons/features.
        output_size : int
            Number of output neurons.
        use_bias : bool, optional
            Whether to use bias in the upper and lower networks (default is True).
        batch_size : int, optional
            Number of samples per batch (default is 32).
        train_weights : bool, optional
            Whether to allow training of weights (default is True).
        train_bias : bool, optional
            Whether to allow training of bias (default is True if `use_bias` is True).
        train_blending : bool, optional
            Whether to allow training of the blending factor (default is False).
        alpha : float, optional
            Scaling parameter for specific activation functions (default depends on activation type).
        lambda_ : Optional[float], optional
            Lambda parameter for `selu` or `elu` activations (default is None).
        train_alpha : bool, optional
            Whether to train the alpha parameter (default is True).
        train_lambda : bool, optional
            Whether to train the lambda parameter (default is True if activation is 'selu' or 'elu').
        activation : str, optional
            Activation function name (default is 'sigmoid'). Options: 'relu', 'tanh', etc.
        weights_uniform_range : Tuple[float, float], optional
            Range for initializing weights uniformly (default is (-1, 1)).
        L2_coe : float, optional
            Coefficient for L2 regularization (default is 0.0).
        L1_coe : float, optional
            Coefficient for L1 regularization (default is 0.0).
        
        Returns:
        -------
        None
        """
        # Define layer dimensions and settings
        self.output_size = output_size  # Number of output neurons
        self.input_size = input_size  # Number of input neurons/features
        self.batch_size = batch_size  # Number of samples in a batch
        self.time_steps = time_steps  # Sequence length for time-series processing
        self.L2_coe = L2_coe  # L2 regularization coefficient
        self.L1_coe = L1_coe  # L1 regularization coefficient
        self.use_bias = use_bias  # Whether biases are used in the computation
        self.activation = activation  # Activation function for the layer

        # Define training flags
        self.train_weights = train_weights  # Allow training of weights
        self.train_bias = False if not use_bias else train_bias  # Only train bias if it's enabled
        self.train_blending = train_blending  # Allow training of blending factors
        self.train_alpha = train_alpha  # Allow training of alpha parameter
        self.train_lambda = False if activation not in ['selu', 'elu'] else train_lambda  # Lambda applicable only for 'selu' or 'elu'

        # Initialize weight ranges for upper and lower networks
        middle = (weights_uniform_range[0] + weights_uniform_range[1]) / 2  # Midpoint of uniform range
        upper_range = (weights_uniform_range[0], middle)  # Upper weight range
        lower_range = (middle, weights_uniform_range[1])  # Lower weight range

        # Weight initialization using a dense initializer function
        self.upper_weight = Dense_weight_init(input_size, output_size, method="uniform", ranges=upper_range)
        self.lower_weight = Dense_weight_init(input_size, output_size, method="uniform", ranges=lower_range)

        # Bias initialization (if enabled)
        if self.use_bias:
            self.upper_bias = np.zeros((output_size, 1))  # Bias for the upper network
            self.lower_bias = np.zeros((output_size, 1))  # Bias for the lower network

        # Blending factor initialization (used for combining upper and lower outputs)
        self.blending_factor = np.zeros((output_size, 1)) + 0.5  # Start blending at a 50-50 weight

        # Alpha initialization (scaling factor for activation functions like 'leaky_relu')
        if alpha is None:
            alpha = 0.01 if activation == 'leaky_relu' else 1.0
        self.alpha = alpha + np.zeros((output_size, 1))  # Broadcast alpha to match output dimensions

        # Lambda parameter initialization for activations like 'selu' and 'elu'
        self.lambda_param = None  # Default to None for unsupported activations
        if activation in ['selu', 'elu']:
            self.lambda_param = (lambda_ if lambda_ is not None else 1.0) + np.zeros((output_size, 1))

        # Network outputs and input storage for forward passes
        self.upper_net = np.zeros((batch_size, time_steps, output_size, 1))  # Upper network output
        self.lower_net = np.zeros((batch_size, time_steps, output_size, 1))  # Lower network output
        self.output = np.zeros((batch_size, time_steps, output_size, 1))  # Combined output
        self.input = np.zeros((batch_size, time_steps, input_size, 1))  # Input storage

        # Gradients initialization for training
        self.grad_w_up = np.zeros(self.upper_weight.shape) if train_weights else None  # Upper weight gradients
        self.grad_w_low = np.zeros(self.lower_weight.shape) if train_weights else None  # Lower weight gradients
        self.grad_bias_up = np.zeros(self.upper_bias.shape) if train_bias else None  # Upper bias gradients
        self.grad_bias_low = np.zeros(self.lower_bias.shape) if train_bias else None  # Lower bias gradients
        self.grad_alpha = np.zeros(self.alpha.shape) if train_alpha else None  # Alpha gradients
        self.grad_lambda = np.zeros(self.lambda_param.shape) if train_lambda else None  # Lambda gradients
        self.grad_blend = np.zeros(self.blending_factor.shape) if train_blending else None  # Blending factor gradients

        #################################################################

    def trainable_params(self) -> int:
        """
        Calculates the number of trainable parameters in the layer.

        Returns:
        --------
        int
            Total count of trainable parameters including weights, biases, 
            alpha, lambda, and blending factors, depending on the training flags.
        """
        params = 0
        if self.train_weights:
            params += np.size(self.upper_weight) * 2  # Both upper and lower weights
        if self.train_bias:
            params += np.size(self.upper_bias) * 2  # Bias for both upper and lower networks
        if self.train_blending:
            params += np.size(self.blending_factor)  # Blending factor
        if self.train_alpha:
            params += np.size(self.alpha)  # Alpha parameter
        if self.train_lambda:
            params += np.size(self.lambda_param)  # Lambda parameter (only for specific activations)
        return params

    #################################################################

    def all_params(self) -> int:
        """
        Calculates the total number of parameters in the layer, including 
        both trainable and non-trainable components.

        Returns:
        --------
        int
            Total count of parameters in the model, including weights, biases, 
            blending factor, alpha, and lambda parameters.
        """
        params = np.size(self.upper_weight) * 2  # Weights for upper and lower networks
        params += np.size(self.blending_factor)  # Blending factor
        params += np.size(self.alpha)  # Alpha parameter
        if self.use_bias:
            params += np.size(self.upper_bias) * 2  # Bias for upper and lower networks
        if self.activation in ['selu', 'elu']:
            params += np.size(self.lambda_param)  # Lambda parameter for specific activations
        return params

    #################################################################

    def __call__(self, batch_index: int, seq_index: int, input: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass through the layer.

        Parameters:
        -----------
        batch_index : int
            Index of the current batch being processed.
        seq_index : int
            Index of the current sequence in the batch.
        input : np.ndarray
            Input data for the layer, expected shape: (input_size,).

        Returns:
        --------
        np.ndarray
            The output of the layer after blending upper and lower network results, 
            shape: (output_size, 1).
        """
        # Store input for forward/backward processing
        self.input[batch_index, seq_index] = input.reshape((-1, 1))

        # Validate batch size
        if self.batch_size < input.shape[0]:
            raise ValueError('Data batch size cannot exceed model batch size')

        # Compute upper and lower network activations
        self.upper_net[batch_index, seq_index] = self.upper_weight @ input.reshape((-1, 1))
        self.lower_net[batch_index, seq_index] = self.lower_weight @ input.reshape((-1, 1))

        # Add biases if enabled
        if self.use_bias:
            self.upper_net[batch_index, seq_index] += self.upper_bias
            self.lower_net[batch_index, seq_index] += self.lower_bias

        # Apply activation functions
        if self.activation in ['selu', 'elu']:
            up_out = net2out(self.upper_net[batch_index, seq_index], self.activation, alpha=self.alpha, lambda_param=self.lambda_param)
            low_out = net2out(self.lower_net[batch_index, seq_index], self.activation, alpha=self.alpha, lambda_param=self.lambda_param)
        else:
            up_out = net2out(self.upper_net[batch_index, seq_index], self.activation, alpha=self.alpha)
            low_out = net2out(self.lower_net[batch_index, seq_index], self.activation, alpha=self.alpha)

        # Compute blending factor
        concat_out = np.concatenate((up_out, low_out), axis=1)
        self.minmax_reverse_stat[batch_index, seq_index] = np.argmax(concat_out)

        # Compute upper and lower outputs
        self.upper_output[batch_index, seq_index] = np.max(concat_out, axis=1).reshape((-1, 1))
        self.lower_output[batch_index, seq_index] = np.min(concat_out, axis=1).reshape((-1, 1))

        # Combine results using the blending factor
        self.output[batch_index, seq_index] = self.blending_factor * self.upper_output[batch_index, seq_index] + \
                                            (1 - self.blending_factor) * self.lower_output[batch_index, seq_index]

        return self.output[batch_index, seq_index]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for training this layer's parameters.

        Parameters:
        -----------
        optimizer : str, optional
            Name of the optimizer (default is 'Adam').
        **kwargs : dict
            Additional arguments for the optimizer configuration.

        Returns:
        --------
        None
        """
        # Set up optimizer based on chosen method and additional parameters
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, batch_size: int, learning_rate: float = 1e-3, grads: np.ndarray = None) -> None:
        """
        Updates the layer's weights, biases, and other trainable parameters
        based on the gradients and learning rate.

        Parameters:
        -----------
        batch_size : int
            Batch size used to average the gradients.
        learning_rate : float, optional
            Learning rate for the parameter updates (default is 1e-3).
        grads : np.ndarray, optional
            Gradient array used to update the parameters. If None, gradients
            are computed internally based on stored values.

        Returns:
        --------
        None
        """
        if grads is None:
            # Add L1 and L2 regularization terms to the gradients for weights
            if self.train_weights:
                self.grad_w_up += self.L1_coe * np.sign(self.upper_weight) + self.L2_coe * self.upper_weight
                self.grad_w_low += self.L1_coe * np.sign(self.lower_weight) + self.L2_coe * self.lower_weight

            # Initialize the gradient vector
            grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))

            if grads is not None:
                # Concatenate gradients for weights, biases, alpha, lambda, and blending factor
                if self.grad_w_up is not None:
                    grads = np.concatenate((grads, self.grad_w_up.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_low.reshape((-1, 1))))
                if self.grad_bias_up is not None:
                    grads = np.concatenate((grads, self.grad_bias_up.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_bias_low.reshape((-1, 1))))
                if self.grad_alpha is not None:
                    grads = np.concatenate((grads, self.grad_alpha.reshape((-1, 1))))
                if self.grad_lambda is not None:
                    grads = np.concatenate((grads, self.grad_lambda.reshape((-1, 1))))
                if self.grad_blend is not None:
                    grads = np.concatenate((grads, self.grad_blend.reshape((-1, 1))))
                
                # Average the gradients over the batch size
                grads /= batch_size

        # Use the optimizer to compute parameter updates (deltas)
        deltas = self.Optimizer(grads, learning_rate)

        # Apply updates to weights, biases, and other parameters
        ind2 = 0
        if self.train_weights:
            # Update upper weights
            ind1 = ind2
            ind2 += int(np.size(self.upper_weight))
            delta_w = deltas[ind1:ind2].reshape(self.upper_weight.shape)
            self.upper_weight -= delta_w

            # Update lower weights
            ind1 = ind2
            ind2 += int(np.size(self.lower_weight))
            delta_w = deltas[ind1:ind2].reshape(self.lower_weight.shape)
            self.lower_weight -= delta_w

        if self.train_bias:
            # Update upper biases
            ind1 = ind2
            ind2 += np.size(self.upper_bias)
            delta_bias = deltas[ind1:ind2].reshape(self.upper_bias.shape)
            self.upper_bias -= delta_bias

            # Update lower biases
            ind1 = ind2
            ind2 += np.size(self.lower_bias)
            delta_bias = deltas[ind1:ind2].reshape(self.lower_bias.shape)
            self.lower_bias -= delta_bias

        if self.train_alpha:
            # Update alpha parameter
            ind1 = ind2
            ind2 += np.size(self.alpha)
            delta_alpha = deltas[ind1:ind2].reshape(self.alpha.shape)
            self.alpha -= delta_alpha

        if self.train_lambda:
            # Update lambda parameter
            ind1 = ind2
            ind2 += np.size(self.lambda_param)
            delta_lambda = deltas[ind1:ind2].reshape(self.lambda_param.shape)
            self.lambda_param -= delta_lambda

        if self.train_blending:
            # Update blending factor
            ind1 = ind2
            ind2 += np.size(self.blending_factor)
            delta_blend = deltas[ind1:ind2].reshape(self.blending_factor.shape)
            self.blending_factor -= delta_blend

        # Reset gradients for the next batch
        self.grad_w_up = np.zeros_like(self.upper_weight) if self.train_weights else None
        self.grad_w_low = np.zeros_like(self.lower_weight) if self.train_weights else None
        self.grad_bias_up = np.zeros_like(self.upper_bias) if self.train_bias else None
        self.grad_bias_low = np.zeros_like(self.lower_bias) if self.train_bias else None
        self.grad_alpha = np.zeros_like(self.alpha) if self.train_alpha else None
        self.grad_lambda = np.zeros_like(self.lambda_param) if self.train_lambda else None
        self.grad_blend = np.zeros_like(self.blending_factor) if self.train_blending else None

    #################################################################

    def return_grads(self):
        """
        Returns the gradients of all trainable parameters in the layer,
        including regularization terms.

        Returns:
        --------
        np.ndarray
            Array of gradients for all trainable parameters.
        """
        # Initialize the gradient array
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))

        if grads is not None:
            # Append weight gradients with regularization
            if self.grad_w_up is not None:
                grad_w_up = self.grad_w_up + self.L1_coe * np.sign(self.upper_weight) + self.L2_coe * self.upper_weight
                grad_w_low = self.grad_w_low + self.L1_coe * np.sign(self.lower_weight) + self.L2_coe * self.lower_weight
                grads = np.concatenate((grads, grad_w_up.reshape((-1, 1))))
                grads = np.concatenate((grads, grad_w_low.reshape((-1, 1))))

            # Append bias gradients
            if self.grad_bias_up is not None:
                grads = np.concatenate((grads, self.grad_bias_up.reshape((-1, 1))))
                grads = np.concatenate((grads, self.grad_bias_low.reshape((-1, 1))))

            # Append alpha gradients
            if self.grad_alpha is not None:
                grads = np.concatenate((grads, self.grad_alpha.reshape((-1, 1))))

            # Append lambda gradients
            if self.grad_lambda is not None:
                grads = np.concatenate((grads, self.grad_lambda.reshape((-1, 1))))

            # Append blending factor gradients
            if self.grad_blend is not None:
                grads = np.concatenate((grads, self.grad_blend.reshape((-1, 1))))

        return grads

    #################################################################

    def backward(self, batch_index: int, seq_index: int, error: np.ndarray) -> np.ndarray:
        """
        Computes gradients for weights, biases, and optional parameters (alpha, lambda, blending factor).
        Propagates error back to the previous layer.

        Parameters:
        -----------
        batch_index : int
            Index of the current batch being processed.
        seq_index : int
            Index of the current sequence in the batch.
        error : np.ndarray
            Error propagated from the subsequent layer, shape should match the layer's output.

        Returns:
        --------
        np.ndarray
            Error propagated to the previous layer, with the same shape as the input.
        """
        # Update gradient for the blending factor if it is trainable
        if self.train_blending:
            self.grad_blend += error.reshape((-1, 1)) * (
                self.upper_output[batch_index, seq_index] - self.lower_output[batch_index, seq_index]
            )

        # Compute the propagated error for the upper and lower networks
        e_max = self.blending_factor * error.reshape((-1, 1))  # Weighted error for the upper network
        e_min = (1 - self.blending_factor) * error.reshape((-1, 1))  # Weighted error for the lower network

        # Allocate errors based on the blending factor and max-min statistics
        e_upper = (
            e_max * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))) +
            e_min * self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))
        )
        e_lower = (
            e_min * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))) +
            e_max * self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))
        )

        # Calculate derivatives for alpha and lambda parameters if required
        if self.train_alpha or self.train_lambda:
            # Compute the derivatives of the activation function w.r.t. alpha and lambda
            Fstar_upper = net2Fstar(self.upper_net[batch_index, seq_index], self.activation, self.alpha, lambda_param=self.lambda_param)
            Fstar_lower = net2Fstar(self.lower_net[batch_index, seq_index], self.activation, self.alpha, lambda_param=self.lambda_param)

            # Update gradients for alpha and lambda if the activation supports them
            if self.activation in ['selu', 'elu']:
                if self.train_alpha:
                    self.grad_alpha += e_upper.reshape((-1, 1)) * Fstar_upper[0]
                    self.grad_alpha += e_lower.reshape((-1, 1)) * Fstar_lower[0]
                if self.train_lambda:
                    self.grad_alpha += e_upper.reshape((-1, 1)) * Fstar_upper[1]
                    self.grad_alpha += e_lower.reshape((-1, 1)) * Fstar_lower[1]
            else:
                if self.train_alpha:
                    self.grad_alpha += e_upper.reshape((-1, 1)) * Fstar_upper
                    self.grad_alpha += e_lower.reshape((-1, 1)) * Fstar_lower

        # Compute the derivative of the activation function for the upper and lower networks
        Fprime_up = net2Fprime(self.upper_net[batch_index, seq_index], self.activation, self.alpha, lambda_param=self.lambda_param)
        Fprime_low = net2Fprime(self.lower_net[batch_index, seq_index], self.activation, self.alpha, lambda_param=self.lambda_param)

        # Calculate the sensitivities (errors) for the weights and biases
        sensitivity_up = e_upper.reshape((-1, 1)) * Fprime_up  # Sensitivity for the upper network
        sensitivity_low = e_lower.reshape((-1, 1)) * Fprime_low  # Sensitivity for the lower network

        # Update gradients for weights if trainable
        if self.train_weights:
            self.grad_w_up += np.outer(sensitivity_up.ravel(), self.input[batch_index, seq_index].ravel())
            self.grad_w_low += np.outer(sensitivity_low.ravel(), self.input[batch_index, seq_index].ravel())

        # Update gradients for biases if trainable
        if self.train_bias:
            self.grad_bias_up += sensitivity_up
            self.grad_bias_low += sensitivity_low

        # Compute the propagated error for the input (previous layer)
        error_in = np.ravel(self.upper_weight.T @ sensitivity_up + self.lower_weight.T @ sensitivity_low)

        # Return the propagated error, reshaped to match the input
        return error_in.reshape((-1, 1))
    
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class FlexibleRough1Feedback:
    """
    Implements a neural network layer-like structure with an additional alpha parameter to control
    the weighting between upper and lower networks. Includes feedback with state vector and 
    separate weight matrices for input and feedback.

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
        Whether to train bias if use_bias or not.
    train_weights : bool, optional
        Whether to train weights or not.
    train_alpha : bool
        Whether to train alpha or not.
    train_lambda : bool
        Whether to train lambda or not.
    train_blending : bool, optional
        If True, blending factor or weight of average will be trained (default is False).
    activation : str, optional
        Activation function to use (default is 'sigmoid'). Other options include 'relu', 'tanh', etc.
    alpha_acti : float, optional
        A scaling parameter for the activation function (default is None).
    weights_uniform_range : tuple, optional
        The range for initializing weights uniformly (default is (-1, 1)).
    L2_coe : float, optional
        L2 regularization coefficient.
    L1_coe : float, optional
        L1 regularization coefficient.
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
                 feedback_size: int = None,
                 alpha: float = None, 
                 lambda_: float = None, 
                 train_alpha: bool = True, 
                 train_lambda: bool = True,
                 activation: str = 'sigmoid', 
                 weights_uniform_range: tuple = (-1, 1),
                 L2_coe: float = 0.0, 
                 L1_coe: float = 0.0):
        """
        Initializes the FlexibleRough1Feedback class with parameters for weights, biases, and feedback.

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
            Whether to train weights or not.
        train_bias : bool, optional
            Whether to train bias if use_bias is enabled (default is True).
        train_blending : bool, optional
            Whether to train blending factor (default is False).
        feedback_size : Optional[int], optional
            Size of the feedback state vector (default is output_size if not specified).
        alpha : Optional[float], optional
            Scaling parameter for activation functions like 'leaky_relu' (default depends on activation type).
        lambda_ : Optional[float], optional
            Lambda parameter for activations like 'selu' or 'elu' (default is None).
        train_alpha : bool
            Whether to train alpha parameter (default is True).
        train_lambda : bool
            Whether to train lambda parameter (default is True for specific activations).
        activation : str, optional
            Activation function to use (default is 'sigmoid').
        weights_uniform_range : Tuple[float, float], optional
            Range for initializing weights uniformly (default is (-1, 1)).
        L2_coe : float, optional
            L2 regularization coefficient (default is 0.0).
        L1_coe : float, optional
            L1 regularization coefficient (default is 0.0).
        """
        self.output_size = output_size  # Number of output neurons.
        self.input_size = input_size  # Number of input features.
        self.batch_size = batch_size  # Number of samples per batch.
        self.time_steps = time_steps  # Number of time steps in the sequence.
        # If feedback size is not provided, set it to the output size.
        self.feedback_size = feedback_size if feedback_size is not None else output_size
        self.L2_coe = L2_coe  # L2 regularization coefficient.
        self.L1_coe = L1_coe  # L1 regularization coefficient.
        self.use_bias = use_bias  # Whether to include bias in computations.
        self.activation = activation  # Name of the activation function.
        self.train_weights = train_weights  # Whether to train weights.
        self.train_bias = False if use_bias is False else train_bias  # Bias training depends on use_bias.
        self.train_blending = train_blending  # Whether to train blending factor.
        self.train_alpha = train_alpha  # Whether to train alpha parameter.
        self.train_lambda = False if (activation != 'selu') and (activation != 'elu') else train_lambda

        # Compute the midpoint of the uniform initialization range.
        middle = (weights_uniform_range[0] + weights_uniform_range[1]) / 2
        upper_range = (weights_uniform_range[0], middle)  # Range for upper weights.
        lower_range = (middle, weights_uniform_range[1])  # Range for lower weights.

        # Initialize upper and lower weights for input and feedback.
        self.upper_weight = Dense_weight_init(input_size, output_size, method="uniform", ranges=upper_range)
        self.lower_weight = Dense_weight_init(input_size, output_size, method="uniform", ranges=lower_range)
        self.upper_weight_state = Dense_weight_init(self.feedback_size, output_size, method="uniform", ranges=upper_range)
        self.lower_weight_state = Dense_weight_init(self.feedback_size, output_size, method="uniform", ranges=lower_range)

        # Initialize biases if use_bias is True.
        if self.use_bias:
            self.upper_bias = np.zeros((output_size, 1))  # Bias for the upper network.
            self.lower_bias = np.zeros((output_size, 1))  # Bias for the lower network.

        # Initialize blending factor (default to 0.5 for equal blending).
        self.blending_factor = np.zeros((output_size, 1)) + 0.5

        # Initialize alpha parameter with a default value if not provided.
        if alpha is None:
            alpha = 0.01 if self.activation == 'leaky_relu' else 1.0
        self.alpha = alpha + np.zeros((output_size, 1))  # Broadcast alpha to match output dimensions.

        # Initialize lambda parameter for specific activation functions.
        self.lambda_param = None
        if self.activation in ['selu', 'elu']:
            self.lambda_param = (lambda_ if lambda_ is not None else 1.0) + np.zeros((output_size, 1))

        # Initialize network outputs for upper and lower networks.
        self.upper_net = np.zeros((batch_size, time_steps, output_size, 1))  # Output from the upper network.
        self.lower_net = np.zeros((batch_size, time_steps, output_size, 1))  # Output from the lower network.

        # Store min-max reverse operation results.
        self.minmax_reverse_stat = np.zeros((batch_size, time_steps, output_size))

        # Final and intermediate outputs.
        self.output = np.zeros((batch_size, time_steps, output_size, 1))  # Final output of the layer.
        self.upper_output = np.zeros((batch_size, time_steps, output_size, 1))  # Upper network output.
        self.lower_output = np.zeros((batch_size, time_steps, output_size, 1))  # Lower network output.
        self.input = np.zeros((batch_size, time_steps, input_size, 1))  # Input storage for the layer.

        # Initialize gradients for weights, biases, alpha, lambda, and blending factor.
        self.grad_w_up = np.zeros(self.upper_weight.shape) if self.train_weights else None
        self.grad_w_low = np.zeros(self.lower_weight.shape) if self.train_weights else None
        self.grad_w_up_state = np.zeros(self.upper_weight_state.shape) if self.train_weights else None
        self.grad_w_low_state = np.zeros(self.lower_weight_state.shape) if self.train_weights else None
        self.grad_bias_low = np.zeros(self.lower_bias.shape) if self.train_bias else None
        self.grad_bias_up = np.zeros(self.upper_bias.shape) if self.train_bias else None
        self.grad_alpha = np.zeros(self.alpha.shape) if self.train_alpha else None
        self.grad_lambda = np.zeros(self.lambda_param.shape) if self.train_lambda else None
        self.grad_blend = np.zeros(self.blending_factor.shape) if self.train_blending else None

    #################################################################

    def trainable_params(self) -> int:
        """
        Calculates the total number of trainable parameters in the model.

        This includes:
        - Upper and lower weights for both input and feedback connections (if trainable).
        - Biases for the upper and lower networks (if enabled and trainable).
        - The blending factor (if trainable).
        - The alpha parameter (if trainable).
        - The lambda parameter (if trainable).

        Returns:
        --------
        int
            The total count of trainable parameters.
        """
        params = 0  # Initialize the parameter counter

        # Count trainable weights for the upper and lower networks
        if self.train_weights:
            # Include weights for input connections
            params += np.size(self.upper_weight) * 2  # Upper and lower weights
            # Include weights for feedback connections
            params += np.size(self.upper_weight_state) * 2  # Upper and lower weights for state feedback

        # Count trainable biases if enabled
        if self.train_bias:
            params += np.size(self.upper_bias) * 2  # Biases for upper and lower networks

        # Count the blending factor if trainable
        if self.train_blending:
            params += np.size(self.blending_factor)  # Blending factor parameter

        # Count the alpha parameter if trainable
        if self.train_alpha:
            params += np.size(self.alpha)  # Alpha parameter

        # Count the lambda parameter if trainable
        if self.train_lambda:
            params += np.size(self.lambda_param)  # Lambda parameter for specific activations

        # Return the total number of trainable parameters as an integer
        return int(params)

    #################################################################

    def all_params(self) -> int:
        """
        Calculates the total number of parameters in the model, including all trainable 
        and non-trainable components.

        This includes:
        - Upper and lower weights for both input and feedback connections.
        - Biases for the upper and lower networks (if enabled).
        - The blending factor.
        - The alpha parameter.
        - The lambda parameter (if applicable for the activation function).

        Returns:
        --------
        int
            The total count of all parameters in the model.
        """
        # Initialize the parameter count with weights for input connections
        params = np.size(self.upper_weight) * 2  # Upper and lower weights for input connections

        # Add weights for feedback connections
        params += np.size(self.upper_weight_state) * 2  # Upper and lower weights for state feedback

        # Add the blending factor parameter
        params += np.size(self.blending_factor)

        # Add the alpha parameter
        params += np.size(self.alpha)

        # Add biases if they are enabled
        if self.use_bias:
            params += np.size(self.upper_bias) * 2  # Biases for upper and lower networks

        # Add the lambda parameter if applicable
        if self.activation in ['selu', 'elu']:
            params += np.size(self.lambda_param)  # Lambda parameter for specific activations

        # Return the total number of parameters as an integer
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
            Input data for the layer, expected to be a 1D vector.
        state : np.ndarray
            State vector representing feedback, also expected to be a 1D vector.

        Returns:
        --------
        np.ndarray
            Output of the layer after processing, returned as a vector.
        """
        # Store the input data for this batch and sequence.
        self.input[batch_index, seq_index] = input.reshape((-1, 1))

        # Ensure the batch size is not exceeded by the input data.
        if self.batch_size < input.shape[0]:
            raise ValueError('Data batch size cannot be larger than model batch size')

        # Compute the upper network's net input by combining input and state contributions.
        self.upper_net[batch_index, seq_index] = (
            self.upper_weight @ input.reshape((-1, 1)) + self.upper_weight_state @ state.reshape((-1, 1))
        )

        # Compute the lower network's net input by combining input and state contributions.
        self.lower_net[batch_index, seq_index] = (
            self.lower_weight @ input.reshape((-1, 1)) + self.lower_weight_state @ state.reshape((-1, 1))
        )

        # Add biases to the upper and lower networks if bias usage is enabled.
        if self.use_bias:
            self.upper_net[batch_index, seq_index] += self.upper_bias
            self.lower_net[batch_index, seq_index] += self.lower_bias

        # Apply the activation function to the net inputs of the upper and lower networks.
        if self.activation in ['selu', 'elu']:
            # Use alpha and lambda parameters for specific activations.
            up_out = net2out(
                self.upper_net[batch_index, seq_index], self.activation, alpha=self.alpha, lambda_param=self.lambda_param
            )
            low_out = net2out(
                self.lower_net[batch_index, seq_index], self.activation, alpha=self.alpha, lambda_param=self.lambda_param
            )
        else:
            # Apply general activation function with alpha.
            up_out = net2out(self.upper_net[batch_index, seq_index], self.activation, alpha=self.alpha)
            low_out = net2out(self.lower_net[batch_index, seq_index], self.activation, alpha=self.alpha)

        # Concatenate the outputs of the upper and lower networks to determine min and max outputs.
        concat_out = np.concatenate((up_out, low_out), axis=1)

        # Store the index of the maximum value for reversing the operation later.
        self.minmax_reverse_stat[batch_index, seq_index] = np.argmax(concat_out)

        # Compute the maximum value for the upper network output.
        self.upper_output[batch_index, seq_index] = np.max(concat_out, axis=1).reshape((-1, 1))

        # Compute the minimum value for the lower network output.
        self.lower_output[batch_index, seq_index] = np.min(concat_out, axis=1).reshape((-1, 1))

        # Combine the upper and lower outputs using the blending factor.
        self.output[batch_index, seq_index] = (
            self.blending_factor * self.upper_output[batch_index, seq_index]
            + (1 - self.blending_factor) * self.lower_output[batch_index, seq_index]
        )

        # Return the final output of the layer for the current batch and sequence.
        return self.output[batch_index, seq_index]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for updating network parameters.

        Parameters:
        -----------
        optimizer : str, optional
            The type of optimizer to use (default is 'Adam').
            Examples: 'Adam', 'SGD', 'RMSprop'.
        **kwargs : dict, optional
            Additional parameters to configure the optimizer, such as learning rate or momentum.

        Returns:
        --------
        None
        """
        # Initialize the optimizer for this layer's parameters.
        # The optimizer uses the total number of trainable parameters.
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, batch_size: int, learning_rate: float = 1e-3, grads: np.ndarray = None) -> None:
        """
        Updates the layer's weights, biases, and other parameters based on calculated gradients.

        Parameters:
        -----------
        batch_size : int
            The batch size used to average gradients.
        learning_rate : float, optional
            Step size for parameter updates (default is 1e-3).
        grads : Optional[np.ndarray], optional
            External gradients for the parameters. If None, gradients are computed internally.

        Returns:
        --------
        None
        """
        if grads is None:
            # If gradients are not provided, compute them internally
            if self.train_weights:
                # Add L1 and L2 regularization terms to weight gradients for input connections
                self.grad_w_up += self.L1_coe * np.sign(self.upper_weight) + self.L2_coe * self.upper_weight
                self.grad_w_low += self.L1_coe * np.sign(self.lower_weight) + self.L2_coe * self.lower_weight
                # Add L1 and L2 regularization terms to weight gradients for state feedback connections
                self.grad_w_up_state += self.L1_coe * np.sign(self.upper_weight_state) + self.L2_coe * self.upper_weight_state
                self.grad_w_low_state += self.L1_coe * np.sign(self.lower_weight_state) + self.L2_coe * self.lower_weight_state

            # Initialize gradient array if there are trainable parameters
            grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))

            # Concatenate all gradients into a single vector
            if grads is not None:
                if self.grad_w_up is not None:
                    grads = np.concatenate((grads, self.grad_w_up.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_low.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_up_state.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_low_state.reshape((-1, 1))))
                if self.grad_bias_up is not None:
                    grads = np.concatenate((grads, self.grad_bias_up.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_bias_low.reshape((-1, 1))))
                if self.grad_alpha is not None:
                    grads = np.concatenate((grads, self.grad_alpha.reshape((-1, 1))))
                if self.grad_lambda is not None:
                    grads = np.concatenate((grads, self.grad_lambda.reshape((-1, 1))))
                if self.grad_blend is not None:
                    grads = np.concatenate((grads, self.grad_blend.reshape((-1, 1))))
                grads /= batch_size  # Average gradients over the batch size

        # Calculate parameter updates using the optimizer
        deltas = self.Optimizer(grads, learning_rate)

        # Initialize an index to track the position within deltas
        ind2 = 0

        # Update weights if trainable
        if self.train_weights:
            # Update input connection weights (upper and lower)
            ind1 = ind2
            ind2 += int(np.size(self.upper_weight))
            delta_w = deltas[ind1:ind2].reshape(self.upper_weight.shape)
            self.upper_weight -= delta_w

            ind1 = ind2
            ind2 += int(np.size(self.lower_weight))
            delta_w = deltas[ind1:ind2].reshape(self.lower_weight.shape)
            self.lower_weight -= delta_w

            # Update feedback connection weights (upper and lower)
            ind1 = ind2
            ind2 += int(np.size(self.upper_weight_state))
            delta_w = deltas[ind1:ind2].reshape(self.upper_weight_state.shape)
            self.upper_weight_state -= delta_w

            ind1 = ind2
            ind2 += int(np.size(self.lower_weight_state))
            delta_w = deltas[ind1:ind2].reshape(self.lower_weight_state.shape)
            self.lower_weight_state -= delta_w

        # Update biases if trainable
        if self.train_bias:
            ind1 = ind2
            ind2 += np.size(self.upper_bias)
            delta_bias = deltas[ind1:ind2].reshape(self.upper_bias.shape)
            self.upper_bias -= delta_bias

            ind1 = ind2
            ind2 += np.size(self.lower_bias)
            delta_bias = deltas[ind1:ind2].reshape(self.lower_bias.shape)
            self.lower_bias -= delta_bias

        # Update alpha parameter if trainable
        if self.train_alpha:
            ind1 = ind2
            ind2 += np.size(self.alpha)
            delta_alpha = deltas[ind1:ind2].reshape(self.alpha.shape)
            self.alpha -= delta_alpha

        # Update lambda parameter if trainable
        if self.train_lambda:
            ind1 = ind2
            ind2 += np.size(self.lambda_param)
            delta_lambda = deltas[ind1:ind2].reshape(self.lambda_param.shape)
            self.lambda_param -= delta_lambda

        # Update blending factor if trainable
        if self.train_blending:
            ind1 = ind2
            ind2 += np.size(self.blending_factor)
            delta_blend = deltas[ind1:ind2].reshape(self.blending_factor.shape)
            self.blending_factor -= delta_blend

        # Reset all gradients for the next iteration
        self.grad_w_up = self.grad_w_up * 0 if self.train_weights else None
        self.grad_w_low = self.grad_w_low * 0 if self.train_weights else None
        self.grad_w_up_state = self.grad_w_up_state * 0 if self.train_weights else None
        self.grad_w_low_state = self.grad_w_low_state * 0 if self.train_weights else None
        self.grad_bias_up = self.grad_bias_up * 0 if self.train_bias else None
        self.grad_bias_low = self.grad_bias_low * 0 if self.train_bias else None
        self.grad_alpha = self.grad_alpha * 0 if self.train_alpha else None
        self.grad_lambda = self.grad_lambda * 0 if self.train_lambda else None
        self.grad_blend = self.grad_blend * 0 if self.train_blending else None

    #################################################################

    def return_grads(self) -> np.ndarray:
        """
        Returns the gradients of all trainable parameters in the layer, including regularization terms.

        This includes:
        - Gradients for upper and lower weights (input and state feedback connections).
        - Gradients for biases.
        - Gradients for alpha, lambda, and blending factors.

        Returns:
        --------
        Optional[np.ndarray]
            A concatenated array of all gradients for trainable parameters.
            Returns None if there are no trainable parameters.
        """
        # Initialize the gradient array if there are trainable parameters
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))

        if grads is not None:
            # Include gradients for weights with regularization
            if self.grad_w_up is not None:
                grad_w_up = self.grad_w_up + (self.L1_coe * np.sign(self.upper_weight) + self.L2_coe * self.upper_weight)
                grad_w_low = self.grad_w_low + (self.L1_coe * np.sign(self.lower_weight) + self.L2_coe * self.lower_weight)
                grad_w_up_state = self.grad_w_up_state + (
                    self.L1_coe * np.sign(self.upper_weight_state) + self.L2_coe * self.upper_weight_state
                )
                grad_w_low_state = self.grad_w_low_state + (
                    self.L1_coe * np.sign(self.lower_weight_state) + self.L2_coe * self.lower_weight_state
                )
                grads = np.concatenate((grads, grad_w_up.reshape((-1, 1))))
                grads = np.concatenate((grads, grad_w_low.reshape((-1, 1))))
                grads = np.concatenate((grads, grad_w_up_state.reshape((-1, 1))))
                grads = np.concatenate((grads, grad_w_low_state.reshape((-1, 1))))

            # Include gradients for biases
            if self.grad_bias_up is not None:
                grads = np.concatenate((grads, self.grad_bias_up.reshape((-1, 1))))
                grads = np.concatenate((grads, self.grad_bias_low.reshape((-1, 1))))

            # Include gradients for alpha
            if self.grad_alpha is not None:
                grads = np.concatenate((grads, self.grad_alpha.reshape((-1, 1))))

            # Include gradients for lambda
            if self.grad_lambda is not None:
                grads = np.concatenate((grads, self.grad_lambda.reshape((-1, 1))))

            # Include gradients for blending factor
            if self.grad_blend is not None:
                grads = np.concatenate((grads, self.grad_blend.reshape((-1, 1))))

        return grads

    #################################################################

    def backward(self, batch_index: int, seq_index: int, error: np.ndarray, state: np.ndarray) -> tuple:
        """
        Computes gradients for weights, biases, and optional parameters (alpha, lambda, blending factor).
        Propagates error back to the previous layer and updates gradients.

        Parameters:
        -----------
        batch_index : int
            Index of the current batch being processed.
        seq_index : int
            Index of the current sequence in the batch.
        error : np.ndarray
            Error propagated from the subsequent layer, expected as a 1D vector.
        state : np.ndarray
            State vector representing feedback, required for updating state-related weights.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - The propagated error for the previous layer input (`error_in`).
            - The propagated error for the state vector (`error_state`).
        """
        # Update gradient for the blending factor if it is trainable
        if self.train_blending:
            self.grad_blend += error.reshape((-1, 1)) * (
                self.upper_output[batch_index, seq_index] - self.lower_output[batch_index, seq_index]
            )

        # Compute the propagated error for the upper and lower networks
        e_max = self.blending_factor * error.reshape((-1, 1))  # Weighted error for the upper network
        e_min = (1 - self.blending_factor) * error.reshape((-1, 1))  # Weighted error for the lower network

        # Allocate errors for upper and lower networks based on the blending factor
        e_upper = (
            e_max * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))) +
            e_min * self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))
        )
        e_lower = (
            e_min * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))) +
            e_max * self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))
        )

        # Calculate derivatives for alpha and lambda parameters if required
        if self.train_alpha or self.train_lambda:
            Fstar_upper = net2Fstar(self.upper_net[batch_index, seq_index], self.activation, self.alpha, lambda_param=self.lambda_param)
            Fstar_lower = net2Fstar(self.lower_net[batch_index, seq_index], self.activation, self.alpha, lambda_param=self.lambda_param)

            if self.activation in ['selu', 'elu']:
                # Update alpha and lambda gradients for specific activations
                if self.train_alpha:
                    self.grad_alpha += e_upper.reshape((-1, 1)) * Fstar_upper[0]
                    self.grad_alpha += e_lower.reshape((-1, 1)) * Fstar_lower[0]
                if self.train_lambda:
                    self.grad_alpha += e_upper.reshape((-1, 1)) * Fstar_upper[1]
                    self.grad_alpha += e_lower.reshape((-1, 1)) * Fstar_lower[1]
            else:
                if self.train_alpha:
                    self.grad_alpha += e_upper.reshape((-1, 1)) * Fstar_upper
                    self.grad_alpha += e_lower.reshape((-1, 1)) * Fstar_lower

        # Compute derivatives of the activation function for upper and lower networks
        Fprime_up = net2Fprime(self.upper_net[batch_index, seq_index], self.activation, self.alpha, lambda_param=self.lambda_param)
        Fprime_low = net2Fprime(self.lower_net[batch_index, seq_index], self.activation, self.alpha, lambda_param=self.lambda_param)

        # Calculate sensitivities for weights and biases
        sensitivity_up = e_upper.reshape((-1, 1)) * Fprime_up
        sensitivity_low = e_lower.reshape((-1, 1)) * Fprime_low

        # Update gradients for weights (input and state connections) if trainable
        if self.train_weights:
            self.grad_w_up += np.outer(sensitivity_up.ravel(), self.input[batch_index, seq_index].ravel())
            self.grad_w_low += np.outer(sensitivity_low.ravel(), self.input[batch_index, seq_index].ravel())
            self.grad_w_up_state += np.outer(sensitivity_up.ravel(), state.ravel())
            self.grad_w_low_state += np.outer(sensitivity_low.ravel(), state.ravel())

        # Update gradients for biases if trainable
        if self.train_bias:
            self.grad_bias_up += sensitivity_up
            self.grad_bias_low += sensitivity_low

        # Compute propagated error for the input (previous layer) and state vector
        error_in = np.ravel(self.upper_weight.T @ sensitivity_up + self.lower_weight.T @ sensitivity_low)
        error_state = np.ravel(self.upper_weight_state.T @ sensitivity_up + self.lower_weight_state.T @ sensitivity_low)

        # Return both propagated errors
        return error_in.reshape((-1, 1)), error_state.reshape((-1, 1))
    
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class FlexibleRough2Feedback:
    """
    Implements a neural network layer-like structure with two feedback mechanisms: Elman and Jordan networks.
    It includes additional alpha parameters to control the weighting between upper and lower networks, with 
    separate weight matrices and biases.

    Parameters:
    ----------
    time_steps : int
        Number of time steps or sequences.
    input_size : int
        Number of input neurons or features.
    output_size : int
        Number of output neurons.
    feedback_size_jordan : int
        Number of rows in the Jordan feedback vector.
    use_bias : bool, optional
        Whether to use bias in the upper and lower networks (default is True).
    batch_size : int, optional
        Number of samples per batch (default is 32).
    train_bias : bool, optional
        Whether to train bias if `use_bias` is enabled (default is True).
    train_weights : bool, optional
        Whether to train weights (default is True).
    train_alpha : bool
        Whether to train alpha parameters (default is True).
    train_lambda : bool
        Whether to train lambda parameters (default is True for specific activations).
    train_blending : bool, optional
        Whether to train the blending factor or weight of the average (default is False).
    activation : str, optional
        Activation function to use (default is 'sigmoid'). Options include 'relu', 'tanh', etc.
    weights_uniform_range : Tuple[float, float], optional
        Range for initializing weights uniformly (default is (-1, 1)).
    L2_coe : float, optional
        L2 regularization coefficient (default is 0.0).
    L1_coe : float, optional
        L1 regularization coefficient (default is 0.0).
    """

    def __init__(
        self,
        time_steps: int,
        input_size: int,
        output_size: int,
        feedback_size_jordan: int,
        use_bias: bool = True,
        batch_size: int = 32,
        train_weights: bool = True,
        train_bias: bool = True,
        train_blending: bool = False,
        alpha: float = None,
        lambda_: float = None,
        train_alpha: bool = True,
        train_lambda: bool = True,
        activation: str = 'sigmoid',
        weights_uniform_range: tuple = (-1, 1),
        L2_coe: float = 0.0,
        L1_coe: float = 0.0,
    ):
        """
        Initializes the `FlexibleRough2Feedback` class with the provided parameters.

        Parameters:
        ----------
        time_steps : int
            Number of time steps or sequences.
        input_size : int
            Number of input neurons or features.
        output_size : int
            Number of output neurons.
        feedback_size_jordan : int
            Number of rows in the Jordan feedback vector.
        use_bias : bool, optional
            Whether to use bias in the upper and lower networks (default is True).
        batch_size : int, optional
            Number of samples per batch (default is 32).
        train_weights : bool, optional
            Whether to train weights (default is True).
        train_bias : bool, optional
            Whether to train bias if `use_bias` is enabled (default is True).
        train_blending : bool, optional
            Whether to train the blending factor or weight of the average (default is False).
        alpha : Optional[float], optional
            Scaling parameter for activation functions like 'leaky_relu' (default depends on activation type).
        lambda_ : Optional[float], optional
            Lambda parameter for activations like 'selu' or 'elu' (default is None).
        train_alpha : bool
            Whether to train alpha parameters (default is True).
        train_lambda : bool
            Whether to train lambda parameters (default is True for specific activations).
        activation : str, optional
            Activation function to use (default is 'sigmoid').
        weights_uniform_range : Tuple[float, float], optional
            Range for initializing weights uniformly (default is (-1, 1)).
        L2_coe : float, optional
            L2 regularization coefficient (default is 0.0).
        L1_coe : float, optional
            L1 regularization coefficient (default is 0.0).
        """
        self.output_size = output_size  # Number of output neurons.
        self.input_size = input_size  # Number of input features.
        self.batch_size = batch_size  # Number of samples per batch.
        self.time_steps = time_steps  # Number of time steps in the sequence.
        self.feedback_size_elman = output_size  # Feedback size for the Elman network.
        self.feedback_size_jordan = feedback_size_jordan  # Feedback size for the Jordan network.
        self.L2_coe = L2_coe  # L2 regularization coefficient.
        self.L1_coe = L1_coe  # L1 regularization coefficient.
        self.use_bias = use_bias  # Whether to include bias in computations.
        self.activation = activation  # Name of the activation function.
        self.train_weights = train_weights  # Whether to train weights.
        self.train_bias = False if not use_bias else train_bias  # Bias training depends on `use_bias`.
        self.train_blending = train_blending  # Whether to train blending factor.
        self.train_alpha = train_alpha  # Whether to train alpha parameter.
        self.train_lambda = train_lambda if activation in ['selu', 'elu'] else False  # Lambda training depends on activation.

        # Compute the midpoint of the uniform initialization range.
        middle = (weights_uniform_range[0] + weights_uniform_range[1]) / 2
        upper_range = (weights_uniform_range[0], middle)  # Range for upper weights.
        lower_range = (middle, weights_uniform_range[1])  # Range for lower weights.

        # Initialize weights for input, Elman feedback, and Jordan feedback connections.
        self.upper_weight = Dense_weight_init(input_size, output_size, method="uniform", ranges=upper_range)
        self.lower_weight = Dense_weight_init(input_size, output_size, method="uniform", ranges=lower_range)
        self.upper_weight_elman = Dense_weight_init(output_size, output_size, method="uniform", ranges=upper_range)
        self.lower_weight_elman = Dense_weight_init(output_size, output_size, method="uniform", ranges=lower_range)
        self.upper_weight_jordan = Dense_weight_init(feedback_size_jordan, output_size, method="uniform", ranges=upper_range)
        self.lower_weight_jordan = Dense_weight_init(feedback_size_jordan, output_size, method="uniform", ranges=lower_range)

        # Initialize biases if `use_bias` is True.
        if self.use_bias:
            self.upper_bias = np.zeros((output_size, 1))  # Bias for the upper network.
            self.lower_bias = np.zeros((output_size, 1))  # Bias for the lower network.

        # Initialize the blending factor (default to 0.5 for equal blending).
        self.blending_factor = np.zeros((output_size, 1)) + 0.5

        # Set the default value for alpha if not provided.
        if alpha is None:
            alpha = 0.01 if activation == 'leaky_relu' else 1.0
        self.alpha = alpha + np.zeros((output_size, 1))  # Ensure alpha matches output dimensions.

        # Initialize lambda for specific activation functions if required.
        self.lambda_param = None
        if self.activation in ['selu', 'elu']:
            self.lambda_param = (lambda_ if lambda_ is not None else 1.0) + np.zeros((output_size, 1))

        # Allocate storage for network outputs and intermediate results.
        self.upper_net = np.zeros((batch_size, time_steps, output_size, 1))  # Upper network output.
        self.lower_net = np.zeros((batch_size, time_steps, output_size, 1))  # Lower network output.
        self.minmax_reverse_stat = np.zeros((batch_size, time_steps, output_size))  # For reversing min-max operations.
        self.output = np.zeros((batch_size, time_steps, output_size, 1))  # Final output.

        # Allocate storage for input and intermediate outputs.
        self.upper_output = np.zeros((batch_size, time_steps, output_size, 1))  # Upper output after activation.
        self.lower_output = np.zeros((batch_size, time_steps, output_size, 1))  # Lower output after activation.
        self.input = np.zeros((batch_size, time_steps, input_size, 1))  # Input storage for the layer.

        # Initialize gradients for weights, biases, and other parameters.
        self.grad_w_up = np.zeros(self.upper_weight.shape) if self.train_weights else None
        self.grad_w_low = np.zeros(self.lower_weight.shape) if self.train_weights else None
        self.grad_w_up_elman = np.zeros(self.upper_weight_elman.shape) if self.train_weights else None
        self.grad_w_low_elman = np.zeros(self.lower_weight_elman.shape) if self.train_weights else None
        self.grad_w_up_jordan = np.zeros(self.upper_weight_jordan.shape) if self.train_weights else None
        self.grad_w_low_jordan = np.zeros(self.lower_weight_jordan.shape) if self.train_weights else None
        self.grad_bias_low = np.zeros(self.lower_bias.shape) if self.train_bias else None
        self.grad_bias_up = np.zeros(self.upper_bias.shape) if self.train_bias else None
        self.grad_alpha = np.zeros(self.alpha.shape) if self.train_alpha else None
        self.grad_lambda = np.zeros(self.lambda_param.shape) if self.train_lambda else None
        self.grad_blend = np.zeros(self.blending_factor.shape) if self.train_blending else None

    #################################################################

    def trainable_params(self) -> int:
        """
        Calculates the total number of trainable parameters in the model.

        This includes:
        - Upper and lower weights for input, Elman feedback, and Jordan feedback connections (if trainable).
        - Biases for the upper and lower networks (if enabled and trainable).
        - The blending factor (if trainable).
        - The alpha parameter (if trainable).
        - The lambda parameter (if trainable).

        Returns:
        --------
        int
            The total count of trainable parameters.
        """
        params = 0  # Initialize the parameter counter

        # Count trainable weights for the upper and lower networks
        if self.train_weights:
            # Include weights for input connections
            params += np.size(self.upper_weight) * 2  # Upper and lower weights
            # Include weights for Elman feedback connections
            params += np.size(self.upper_weight_elman) * 2  # Upper and lower weights for Elman feedback
            # Include weights for Jordan feedback connections
            params += np.size(self.upper_weight_jordan) * 2  # Upper and lower weights for Jordan feedback

        # Count trainable biases if enabled
        if self.train_bias:
            params += np.size(self.upper_bias) * 2  # Biases for upper and lower networks

        # Count the blending factor if trainable
        if self.train_blending:
            params += np.size(self.blending_factor)  # Blending factor parameter

        # Count the alpha parameter if trainable
        if self.train_alpha:
            params += np.size(self.alpha)  # Alpha parameter

        # Count the lambda parameter if trainable
        if self.train_lambda:
            params += np.size(self.lambda_param)  # Lambda parameter for specific activations

        # Return the total number of trainable parameters as an integer
        return int(params)

    #################################################################

    def all_params(self) -> int:
        """
        Calculates the total number of parameters in the model, including all trainable 
        and non-trainable components.

        This includes:
        - Upper and lower weights for input, Elman feedback, and Jordan feedback connections.
        - Biases for the upper and lower networks (if enabled).
        - The blending factor.
        - The alpha parameter.
        - The lambda parameter (if applicable for the activation function).

        Returns:
        --------
        int
            The total count of all parameters in the model.
        """
        # Initialize the parameter count with weights for input connections
        params = np.size(self.upper_weight) * 2  # Upper and lower weights for input connections

        # Add weights for Elman feedback connections
        params += np.size(self.upper_weight_elman) * 2  # Upper and lower weights for Elman feedback

        # Add weights for Jordan feedback connections
        params += np.size(self.upper_weight_jordan) * 2  # Upper and lower weights for Jordan feedback

        # Add the blending factor parameter
        params += np.size(self.blending_factor)

        # Add the alpha parameter
        params += np.size(self.alpha)

        # Add biases if they are enabled
        if self.use_bias:
            params += np.size(self.upper_bias) * 2  # Biases for upper and lower networks

        # Add the lambda parameter if applicable
        if self.activation in ['selu', 'elu']:
            params += np.size(self.lambda_param)  # Lambda parameter for specific activations

        # Return the total number of parameters as an integer
        return int(params)

    #################################################################

    def __call__(
        self,
        batch_index: int,
        seq_index: int,
        input: np.ndarray,
        elman_state: np.ndarray,
        jordan_state: np.ndarray
    ) -> np.ndarray:
        """
        Perform the forward pass of the model.

        Parameters:
        -----------
        batch_index : int
            Index of the current batch being processed.
        seq_index : int
            Index of the current sequence in the batch.
        input : np.ndarray
            Input data for the layer, expected to be a 1D vector.
        elman_state : np.ndarray
            Elman feedback state vector from this layer, expected to be a 1D vector.
        jordan_state : np.ndarray
            Jordan feedback state vector from the previous layer, expected to be a 1D vector.

        Returns:
        --------
        np.ndarray
            The final output of the layer, returned as a 1D vector.
        """
        # Store the input data for the current batch and sequence
        self.input[batch_index, seq_index] = input.reshape((-1, 1))

        # Ensure that the batch size matches the input's batch size
        if self.batch_size < input.shape[0]:
            raise ValueError('Data batch size cannot be larger than model batch size')

        # Compute the net input for the upper network by combining input and feedback contributions
        self.upper_net[batch_index, seq_index] = (
            self.upper_weight @ input.reshape((-1, 1)) +
            self.upper_weight_elman @ elman_state.reshape((-1, 1)) +
            self.upper_weight_jordan @ jordan_state.reshape((-1, 1))
        )

        # Compute the net input for the lower network by combining input and feedback contributions
        self.lower_net[batch_index, seq_index] = (
            self.lower_weight @ input.reshape((-1, 1)) +
            self.lower_weight_elman @ elman_state.reshape((-1, 1)) +
            self.lower_weight_jordan @ jordan_state.reshape((-1, 1))
        )

        # Add biases to the upper and lower networks if bias usage is enabled
        if self.use_bias:
            self.upper_net[batch_index, seq_index] += self.upper_bias
            self.lower_net[batch_index, seq_index] += self.lower_bias

        # Apply the activation function to the net inputs for the upper and lower networks
        if self.activation in ['selu', 'elu']:
            # Use alpha and lambda parameters for specific activations
            up_out = net2out(
                self.upper_net[batch_index, seq_index], self.activation, alpha=self.alpha, lambda_param=self.lambda_param
            )
            low_out = net2out(
                self.lower_net[batch_index, seq_index], self.activation, alpha=self.alpha, lambda_param=self.lambda_param
            )
        else:
            # Apply general activation function with alpha
            up_out = net2out(self.upper_net[batch_index, seq_index], self.activation, alpha=self.alpha)
            low_out = net2out(self.lower_net[batch_index, seq_index], self.activation, alpha=self.alpha)

        # Concatenate the outputs of the upper and lower networks to determine min and max values
        concat_out = np.concatenate((up_out, low_out), axis=1)

        # Store the index of the maximum value for reversing the operation later
        self.minmax_reverse_stat[batch_index, seq_index] = np.argmax(concat_out)

        # Compute the maximum value for the upper network output
        self.upper_output[batch_index, seq_index] = np.max(concat_out, axis=1).reshape((-1, 1))

        # Compute the minimum value for the lower network output
        self.lower_output[batch_index, seq_index] = np.min(concat_out, axis=1).reshape((-1, 1))

        # Combine the upper and lower outputs using the blending factor
        self.output[batch_index, seq_index] = (
            self.blending_factor * self.upper_output[batch_index, seq_index] +
            (1 - self.blending_factor) * self.lower_output[batch_index, seq_index]
        )

        # Return the final output of the layer for the current batch and sequence
        return self.output[batch_index, seq_index]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for updating network parameters.

        Parameters:
        -----------
        optimizer : str, optional
            The type of optimizer to use (default is 'Adam').
            Examples include 'Adam', 'SGD', 'RMSprop'.
        **kwargs : dict, optional
            Additional parameters for configuring the optimizer, such as learning rate or momentum.

        Returns:
        --------
        None
        """
        # Initialize the optimizer for this layer's parameters
        # The optimizer uses the total number of trainable parameters in the layer
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, batch_size: int, learning_rate: float = 1e-3, grads: np.ndarray = None) -> None:
        """
        Updates the layer's weights, biases, and other parameters based on calculated gradients.

        Parameters:
        -----------
        batch_size : int
            The batch size used to average gradients.
        learning_rate : float, optional
            Step size for parameter updates (default is 1e-3).
        grads : Optional[np.ndarray], optional
            External gradients for the parameters. If None, gradients are computed internally.

        Returns:
        --------
        None
        """
        # If external gradients are not provided, compute them internally
        if grads is None:
            if self.train_weights:
                # Add L1 and L2 regularization terms to weight gradients for input connections
                self.grad_w_up += self.L1_coe * np.sign(self.upper_weight) + self.L2_coe * self.upper_weight
                self.grad_w_low += self.L1_coe * np.sign(self.lower_weight) + self.L2_coe * self.lower_weight
                # Add regularization terms to Elman feedback weight gradients
                self.grad_w_up_elman += self.L1_coe * np.sign(self.upper_weight_elman) + self.L2_coe * self.upper_weight_elman
                self.grad_w_low_elman += self.L1_coe * np.sign(self.lower_weight_elman) + self.L2_coe * self.lower_weight_elman
                # Add regularization terms to Jordan feedback weight gradients
                self.grad_w_up_jordan += self.L1_coe * np.sign(self.upper_weight_jordan) + self.L2_coe * self.upper_weight_jordan
                self.grad_w_low_jordan += self.L1_coe * np.sign(self.lower_weight_jordan) + self.L2_coe * self.lower_weight_jordan

            # Initialize gradient array if there are trainable parameters
            grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))

            # Concatenate all gradients into a single vector
            if grads is not None:
                if self.grad_w_up is not None:
                    grads = np.concatenate((grads, self.grad_w_up.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_low.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_up_elman.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_low_elman.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_up_jordan.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_low_jordan.reshape((-1, 1))))
                if self.grad_bias_up is not None:
                    grads = np.concatenate((grads, self.grad_bias_up.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_bias_low.reshape((-1, 1))))
                if self.grad_alpha is not None:
                    grads = np.concatenate((grads, self.grad_alpha.reshape((-1, 1))))
                if self.grad_lambda is not None:
                    grads = np.concatenate((grads, self.grad_lambda.reshape((-1, 1))))
                if self.grad_blend is not None:
                    grads = np.concatenate((grads, self.grad_blend.reshape((-1, 1))))
                grads /= batch_size  # Average gradients over the batch size

        # Calculate parameter updates using the optimizer
        deltas = self.Optimizer(grads, learning_rate)

        # Initialize an index to track the position within deltas
        ind2 = 0

        # Update weights if trainable
        if self.train_weights:
            # Update input connection weights (upper and lower)
            ind1 = ind2
            ind2 += int(np.size(self.upper_weight))
            delta_w = deltas[ind1:ind2].reshape(self.upper_weight.shape)
            self.upper_weight -= delta_w

            ind1 = ind2
            ind2 += int(np.size(self.lower_weight))
            delta_w = deltas[ind1:ind2].reshape(self.lower_weight.shape)
            self.lower_weight -= delta_w

            # Update Elman feedback connection weights (upper and lower)
            ind1 = ind2
            ind2 += int(np.size(self.upper_weight_elman))
            delta_w = deltas[ind1:ind2].reshape(self.upper_weight_elman.shape)
            self.upper_weight_elman -= delta_w

            ind1 = ind2
            ind2 += int(np.size(self.lower_weight_elman))
            delta_w = deltas[ind1:ind2].reshape(self.lower_weight_elman.shape)
            self.lower_weight_elman -= delta_w

            # Update Jordan feedback connection weights (upper and lower)
            ind1 = ind2
            ind2 += int(np.size(self.upper_weight_jordan))
            delta_w = deltas[ind1:ind2].reshape(self.upper_weight_jordan.shape)
            self.upper_weight_jordan -= delta_w

            ind1 = ind2
            ind2 += int(np.size(self.lower_weight_jordan))
            delta_w = deltas[ind1:ind2].reshape(self.lower_weight_jordan.shape)
            self.lower_weight_jordan -= delta_w

        # Update biases if trainable
        if self.train_bias:
            ind1 = ind2
            ind2 += np.size(self.upper_bias)
            delta_bias = deltas[ind1:ind2].reshape(self.upper_bias.shape)
            self.upper_bias -= delta_bias

            ind1 = ind2
            ind2 += np.size(self.lower_bias)
            delta_bias = deltas[ind1:ind2].reshape(self.lower_bias.shape)
            self.lower_bias -= delta_bias

        # Update alpha parameter if trainable
        if self.train_alpha:
            ind1 = ind2
            ind2 += np.size(self.alpha)
            delta_alpha = deltas[ind1:ind2].reshape(self.alpha.shape)
            self.alpha -= delta_alpha

        # Update lambda parameter if trainable
        if self.train_lambda:
            ind1 = ind2
            ind2 += np.size(self.lambda_param)
            delta_lambda = deltas[ind1:ind2].reshape(self.lambda_param.shape)
            self.lambda_param -= delta_lambda

        # Update blending factor if trainable
        if self.train_blending:
            ind1 = ind2
            ind2 += np.size(self.blending_factor)
            delta_blend = deltas[ind1:ind2].reshape(self.blending_factor.shape)
            self.blending_factor -= delta_blend

        # Reset all gradients for the next iteration
        self.grad_w_up = self.grad_w_up * 0 if self.train_weights else None
        self.grad_w_low = self.grad_w_low * 0 if self.train_weights else None
        self.grad_w_up_elman = self.grad_w_up_elman * 0 if self.train_weights else None
        self.grad_w_low_elman = self.grad_w_low_elman * 0 if self.train_weights else None
        self.grad_w_up_jordan = self.grad_w_up_jordan * 0 if self.train_weights else None
        self.grad_w_low_jordan = self.grad_w_low_jordan * 0 if self.train_weights else None
        self.grad_bias_up = self.grad_bias_up * 0 if self.train_bias else None
        self.grad_bias_low = self.grad_bias_low * 0 if self.train_bias else None
        self.grad_alpha = self.grad_alpha * 0 if self.train_alpha else None
        self.grad_lambda = self.grad_lambda * 0 if self.train_lambda else None
        self.grad_blend = self.grad_blend * 0 if self.train_blending else None

    #################################################################

    def return_grads(self) -> np.ndarray:
        """
        Returns the gradients of all trainable parameters in the layer, including regularization terms.

        This includes:
        - Gradients for upper and lower weights (input, Elman feedback, and Jordan feedback connections).
        - Gradients for biases.
        - Gradients for alpha, lambda, and blending factors.

        Returns:
        --------
        Optional[np.ndarray]
            A concatenated array of all gradients for trainable parameters.
            Returns None if there are no trainable parameters.
        """
        # Initialize the gradient array if there are trainable parameters
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))

        if grads is not None:
            # Include gradients for weights with regularization
            if self.grad_w_up is not None:
                grad_w_up = self.grad_w_up + (self.L1_coe * np.sign(self.upper_weight) + self.L2_coe * self.upper_weight)
                grad_w_low = self.grad_w_low + (self.L1_coe * np.sign(self.lower_weight) + self.L2_coe * self.lower_weight)
                grad_w_up_elman = self.grad_w_up_elman + (
                    self.L1_coe * np.sign(self.upper_weight_elman) + self.L2_coe * self.upper_weight_elman
                )
                grad_w_low_elman = self.grad_w_low_elman + (
                    self.L1_coe * np.sign(self.lower_weight_elman) + self.L2_coe * self.lower_weight_elman
                )
                grad_w_up_jordan = self.grad_w_up_jordan + (
                    self.L1_coe * np.sign(self.upper_weight_jordan) + self.L2_coe * self.upper_weight_jordan
                )
                grad_w_low_jordan = self.grad_w_low_jordan + (
                    self.L1_coe * np.sign(self.lower_weight_jordan) + self.L2_coe * self.lower_weight_jordan
                )
                grads = np.concatenate((grads, grad_w_up.reshape((-1, 1))))
                grads = np.concatenate((grads, grad_w_low.reshape((-1, 1))))
                grads = np.concatenate((grads, grad_w_up_elman.reshape((-1, 1))))
                grads = np.concatenate((grads, grad_w_low_elman.reshape((-1, 1))))
                grads = np.concatenate((grads, grad_w_up_jordan.reshape((-1, 1))))
                grads = np.concatenate((grads, grad_w_low_jordan.reshape((-1, 1))))

            # Include gradients for biases
            if self.grad_bias_up is not None:
                grads = np.concatenate((grads, self.grad_bias_up.reshape((-1, 1))))
                grads = np.concatenate((grads, self.grad_bias_low.reshape((-1, 1))))

            # Include gradients for alpha
            if self.grad_alpha is not None:
                grads = np.concatenate((grads, self.grad_alpha.reshape((-1, 1))))

            # Include gradients for lambda
            if self.grad_lambda is not None:
                grads = np.concatenate((grads, self.grad_lambda.reshape((-1, 1))))

            # Include gradients for blending factor
            if self.grad_blend is not None:
                grads = np.concatenate((grads, self.grad_blend.reshape((-1, 1))))

        return grads

    #################################################################

    def backward(
        self,
        batch_index: int,
        seq_index: int,
        error: np.ndarray,
        elman_state: np.ndarray,
        jordan_state: np.ndarray
    ) -> tuple:
        """
        Computes gradients for weights, biases, and other parameters (alpha, lambda, blending factor).
        Propagates error back to the previous layer and updates gradients.

        Parameters:
        -----------
        batch_index : int
            Index of the current batch being processed.
        seq_index : int
            Index of the current sequence in the batch.
        error : np.ndarray
            Error propagated from the subsequent layer, expected as a 1D vector.
        elman_state : np.ndarray
            Elman feedback state vector from this layer, expected as a 1D vector.
        jordan_state : np.ndarray
            Jordan feedback state vector from the previous layer, expected as a 1D vector.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing:
            - The propagated error for the input (`error_in`).
            - The propagated error for the Elman feedback state (`error_elman`).
            - The propagated error for the Jordan feedback state (`error_jordan`).
        """
        # Update gradient for the blending factor if it is trainable
        if self.train_blending:
            self.grad_blend += error.reshape((-1, 1)) * (
                self.upper_output[batch_index, seq_index] - self.lower_output[batch_index, seq_index]
            )

        # Compute the propagated error for the upper and lower networks
        e_max = self.blending_factor * error.reshape((-1, 1))  # Weighted error for the upper network
        e_min = (1 - self.blending_factor) * error.reshape((-1, 1))  # Weighted error for the lower network

        # Allocate errors for upper and lower networks based on the blending factor
        e_upper = (
            e_max * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))) +
            e_min * self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))
        )
        e_lower = (
            e_min * np.logical_not(self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))) +
            e_max * self.minmax_reverse_stat[batch_index, seq_index].reshape((-1, 1))
        )

        # Calculate derivatives for alpha and lambda parameters if required
        if self.train_alpha or self.train_lambda:
            Fstar_upper = net2Fstar(self.upper_net[batch_index, seq_index], self.activation, self.alpha, lambda_param=self.lambda_param)
            Fstar_lower = net2Fstar(self.lower_net[batch_index, seq_index], self.activation, self.alpha, lambda_param=self.lambda_param)

            if self.activation in ['selu', 'elu']:
                # Update alpha and lambda gradients for specific activations
                if self.train_alpha:
                    self.grad_alpha += e_upper.reshape((-1, 1)) * Fstar_upper[0]
                    self.grad_alpha += e_lower.reshape((-1, 1)) * Fstar_lower[0]
                if self.train_lambda:
                    self.grad_alpha += e_upper.reshape((-1, 1)) * Fstar_upper[1]
                    self.grad_alpha += e_lower.reshape((-1, 1)) * Fstar_lower[1]
            else:
                if self.train_alpha:
                    self.grad_alpha += e_upper.reshape((-1, 1)) * Fstar_upper
                    self.grad_alpha += e_lower.reshape((-1, 1)) * Fstar_lower

        # Compute derivatives of the activation function for upper and lower networks
        Fprime_up = net2Fprime(self.upper_net[batch_index, seq_index], self.activation, self.alpha, lambda_param=self.lambda_param)
        Fprime_low = net2Fprime(self.lower_net[batch_index, seq_index], self.activation, self.alpha, lambda_param=self.lambda_param)

        # Calculate sensitivities for weights and biases
        sensitivity_up = e_upper.reshape((-1, 1)) * Fprime_up
        sensitivity_low = e_lower.reshape((-1, 1)) * Fprime_low

        # Update gradients for weights (input, Elman feedback, and Jordan feedback connections) if trainable
        if self.train_weights:
            self.grad_w_up += np.outer(sensitivity_up.ravel(), self.input[batch_index, seq_index].ravel())
            self.grad_w_low += np.outer(sensitivity_low.ravel(), self.input[batch_index, seq_index].ravel())
            self.grad_w_up_elman += np.outer(sensitivity_up.ravel(), elman_state.ravel())
            self.grad_w_low_elman += np.outer(sensitivity_low.ravel(), elman_state.ravel())
            self.grad_w_up_jordan += np.outer(sensitivity_up.ravel(), jordan_state.ravel())
            self.grad_w_low_jordan += np.outer(sensitivity_low.ravel(), jordan_state.ravel())

        # Update gradients for biases if trainable
        if self.train_bias:
            self.grad_bias_up += sensitivity_up
            self.grad_bias_low += sensitivity_low

        # Compute propagated error for the input, Elman feedback, and Jordan feedback
        error_in = np.ravel(self.upper_weight.T @ sensitivity_up + self.lower_weight.T @ sensitivity_low)
        error_elman = np.ravel(self.upper_weight_elman.T @ sensitivity_up + self.lower_weight_elman.T @ sensitivity_low)
        error_jordan = np.ravel(self.upper_weight_jordan.T @ sensitivity_up + self.lower_weight_jordan.T @ sensitivity_low)

        # Return all propagated errors
        return error_in.reshape((-1, 1)), error_elman.reshape((-1, 1)), error_jordan.reshape((-1, 1))
