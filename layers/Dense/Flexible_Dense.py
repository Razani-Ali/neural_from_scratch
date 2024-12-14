import numpy as np
from activations.flexible_activation_functions import net2out, net2Fprime, net2Fstar
from initializers.weight_initializer import Dense_weight_init
from optimizers.set_optimizer import init_optimizer


class FlexibleDense:
    """
    A flexible Dense layer implementation that supports various activation functions, initialization options,
    and batch processing.

    Parameters:
    - input_size (int): Number of input features.
    - output_size (int): Number of neurons in the layer.
    - use_bias (bool): Whether to include a bias term. Default is True.
    - batch_size (int): Number of samples per batch. Default is 32.
    - activation (str): Activation function to use. Options are 'leaky_relu', 'selu', 'elu', 'sigmoid', 'tanh'.
    - alpha (float, optional): Parameter for activation functions that require an alpha value.
    - lambda_ (float, optional): Scaling factor, required for 'selu' and 'elu'.
    - train_weights (bool): Whether to train weights or not
    - train_bias (bool): Whether to train bias or not
    - train_alpha (bool): Whether to train alpha or not
    - train_lambda (bool): Whether to train lambda or not
    - weights_init_method (str): Weight initialization method (e.g., 'he' or 'xavier').
    - weight_distribution (str): Distribution of weights ('normal' or 'uniform').
    - orthogonal_scale_factor (float): Scale factor for orthogonal initialization.
    - weights_uniform_range (tuple, optional): Range for uniform weight distribution.
    - L2_coe (float, optional): L2 regularization coefficient
    - L1_coe (float, optional): L1 regularization coefficient

    Attributes:
    - weight (np.ndarray): Weight matrix of shape (output_size, input_size).
    - bias (np.ndarray): Bias vector of shape (output_size, 1), initialized to zero if `use_bias` is True.
    - alpha (np.ndarray): Alpha parameter for activation, shaped (output_size, 1).
    - lambda_param (np.ndarray): Lambda parameter for SELU or ELU activation, if applicable.
    - net (np.ndarray): Pre-activation values for each neuron in each batch.
    - output (np.ndarray): Activated output values for each neuron in each batch.
    """

    def __init__(self, input_size: int, output_size: int, use_bias: bool = True, batch_size: int = 32, 
                 activation: str = 'leaky_relu', alpha: float = None, lambda_=None,
                 train_weights: bool = True, train_bias: bool = True, train_alpha: bool = True, train_lambda: bool = True,
                 weights_init_method: str = 'he', L2_coe: float = 0.0, L1_coe: float = 0.0,
                 weight_distribution: str = 'normal', orthogonal_scale_factor: float = 1.0, 
                 weights_uniform_range: tuple = None):
        self.output_size = output_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.L2_coe = L2_coe
        self.L1_coe = L1_coe
        self.use_bias = use_bias
        self.activation = activation
        self.train_weights = train_weights
        self.train_alpha = train_alpha
        self.train_bias = False if use_bias is False else train_bias
        self.train_lambda = False if (activation != 'selu') and (activation != 'elu') else train_lambda

        # Initialize weights based on the chosen initialization method and distribution
        self.weight = Dense_weight_init(input_size, output_size, method=weights_init_method, 
                                        distribution=weight_distribution, scale_factor=orthogonal_scale_factor, 
                                        ranges=weights_uniform_range)
        
        # Initialize bias if enabled
        if self.use_bias:
            self.bias = np.zeros((output_size, 1))
        
        # Set default alpha values if not provided
        if alpha is None:
            alpha = 0.01 if self.activation == 'leaky_relu' else 1.0
        self.alpha = alpha + np.zeros((output_size, 1))  # Ensure alpha has correct shape
        
        # Set default lambda values for SELU or ELU activation
        self.lambda_param = None
        if self.activation in ['selu', 'elu']:
            self.lambda_param = (lambda_ if lambda_ is not None else 1.0) + np.zeros((output_size, 1))

        # Initialize pre-activation and output arrays for batch processing
        self.net = np.zeros((batch_size, output_size, 1))
        self.output = np.zeros((batch_size, output_size, 1))

    #################################################################

    def trainable_params(self) -> int:
        """
        Returns the total number of trainable parameters in the layer.

        Returns:
        - int: Number of trainable parameters.
        """
        params = 0
        if self.train_weights:
            params += np.size(self.weight)
        if self.train_bias:
            params += np.size(self.bias)
        if self.train_alpha:
            params += np.size(self.alpha)
        if self.train_lambda:
            params += np.size(self.lambda_param)
        return params
    
    #################################################################

    def all_params(self) -> int:
        """
        Returns the total number of parameters in the layer.

        Returns:
        - int: Number of parameters.
        """
        params = np.size(self.alpha) + np.size(self.weight)
        if self.use_bias:
            params += np.size(self.bias)
        if self.activation in ['selu', 'elu']:
            params += np.size(self.lambda_param)
        return params

    #################################################################

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Applies the Dense layer to the input.

        Parameters:
        - input (np.ndarray): Input matrix of shape (batch_size, input_size).

        Returns:
        - np.ndarray: Activated output of the layer, shape (batch_size, output_size).
        """
        self.input = input

        # Check batch size consistency
        if self.batch_size < input.shape[0]:
            raise ValueError('Data batch size cannot be larger than model batch size')

        # Process each sample in the batch
        for batch_index, input_vector in enumerate(input):
            # Linear transformation
            self.net[batch_index] = self.weight @ input_vector.reshape((-1, 1))
            if self.use_bias:
                self.net[batch_index] += self.bias
            # Apply activation function
            self.output[batch_index] = net2out(self.net[batch_index], self.activation, alpha=self.alpha, lambda_param=self.lambda_param)

        batch_index += 1
        return self.output[:batch_index, :, 0]

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
        # Initialize optimizer with given parameters and assign to layer's optimizer attribute
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

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
            ind2 += np.size(self.alpha)  # End index for alpha
            delta_alpha = deltas[ind1:ind2].reshape(self.alpha.shape)
            self.alpha -= delta_alpha  # Update alpha

        # Update lambda parameter if trainable
        if self.train_lambda:
            ind1 = ind2  # Start index for lambda
            ind2 += np.size(self.lambda_param)  # End index for lambda
            delta_lambda = deltas[ind1:ind2].reshape(self.lambda_param.shape)
            self.lambda_param -= delta_lambda  # Update lambda

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
            Returns dictionary with `error_in` and `gradients` if both `return_error` and `return_grads` are True.
            Returns `error_in` if `return_error` is True and `return_grads` is False.
            Returns `gradients` if `return_grads` is True and `return_error` is False.
        """
        # Initialize error gradient for inputs if return_error is True
        if return_error:
            error_in = np.zeros(self.input.shape)

        # Initialize gradients for weights, biases, alpha, and lambda parameters
        grad_w = np.zeros(self.weight.shape) if self.train_weights else None
        grad_bias = np.zeros(self.bias.shape) if self.train_bias else None
        grad_alpha = np.zeros(self.alpha.shape) if self.train_alpha else None
        grad_lambda = np.zeros(self.lambda_param.shape) if self.train_lambda else None

        # Process each error in the batch
        for batch_index, one_batch_error in enumerate(error_batch):
            one_batch_error = one_batch_error.reshape((-1,1))

            # Calculate derivatives for alpha and lambda if required
            if self.train_alpha or self.train_lambda:
                Fstar = net2Fstar(self.net[batch_index], self.activation, self.alpha, lambda_param=self.lambda_param)
                # Update alpha and lambda gradients based on activation function
                if self.activation in ['selu', 'elu']:
                    if self.train_alpha:
                        grad_alpha += one_batch_error * Fstar[0]
                    if self.train_lambda:
                        grad_lambda += one_batch_error * Fstar[1]
                else:
                    if self.train_alpha:
                        grad_alpha += one_batch_error * Fstar

            # Compute derivative of the activation function
            Fprime = net2Fprime(self.net[batch_index], self.activation, self.alpha, lambda_param=self.lambda_param)

            # Calculate sensitivity as the product of the error and activation derivative
            sensitivity = one_batch_error * Fprime

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
            grad_alpha /= error_batch.shape[0]
        if self.train_bias:
            grad_bias /= error_batch.shape[0]
        if self.train_lambda:
            grad_lambda /= error_batch.shape[0]

        # Collect gradients into a single array if not None
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1,1))
        if grads is not None:
            if grad_w is not None:
                grads = np.concatenate((grads, grad_w.reshape((-1,1))))
            if grad_bias is not None:
                grads = np.concatenate((grads, grad_bias.reshape((-1,1))))
            if grad_alpha is not None:
                grads = np.concatenate((grads, grad_alpha.reshape((-1,1))))
            if grad_lambda is not None:
                grads = np.concatenate((grads, grad_lambda.reshape((-1,1))))

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

class TimeFlexibleDense:
    """
    A flexible Dense layer that supports batch processing and various activation functions.

    Parameters:
        time_steps (int): Number of time steps in the sequence data.
        input_size (int): Number of input features per time step.
        output_size (int): Number of neurons in the dense layer.
        use_bias (bool): Whether to include a bias term in the computation. Default is True.
        batch_size (int): Number of samples per batch. Default is 32.
        activation (str): Activation function to use (e.g., 'leaky_relu', 'selu', 'elu', 'sigmoid', 'tanh').
        alpha (float, optional): Parameter for activation functions (e.g., slope in 'leaky_relu').
        lambda_ (float, optional): Scaling factor for activations like 'selu' or 'elu'.
        train_weights (bool): Whether to allow training of weights. Default is True.
        train_bias (bool): Whether to allow training of biases. Default is True.
        train_alpha (bool): Whether to allow training of the alpha parameter. Default is True.
        train_lambda (bool): Whether to allow training of the lambda parameter. Default is True.
        weights_init_method (str): Initialization method for weights (e.g., 'he', 'xavier').
        L2_coe (float, optional): L2 regularization coefficient. Default is 0.0.
        L1_coe (float, optional): L1 regularization coefficient. Default is 0.0.
        weight_distribution (str): Distribution for weight initialization ('normal' or 'uniform').
        orthogonal_scale_factor (float): Scale factor for orthogonal initialization.
        weights_uniform_range (tuple, optional): Range for uniform weight distribution.

    Attributes:
        weight (np.ndarray): Weight matrix of shape (output_size, input_size).
        bias (np.ndarray): Bias vector of shape (output_size, 1), initialized to zero if `use_bias` is True.
        alpha (np.ndarray): Alpha parameter for activation, shaped (output_size, 1).
        lambda_param (np.ndarray): Lambda parameter for SELU or ELU activation, if applicable.
        net (np.ndarray): Pre-activation values for each neuron in each batch.
        output (np.ndarray): Activated output values for each neuron in each batch.
    """
    
    def __init__(self, time_steps: int, input_size: int, output_size: int, use_bias: bool = True,
                 batch_size: int = 32, activation: str = 'leaky_relu', alpha: float = None, lambda_=None,
                 train_weights: bool = True, train_bias: bool = True, train_alpha: bool = True, train_lambda: bool = True,
                 weights_init_method: str = 'he', L2_coe: float = 0.0, L1_coe: float = 0.0,
                 weight_distribution: str = 'normal', orthogonal_scale_factor: float = 1.0,
                 weights_uniform_range: tuple = None):
        """
        Initializes the TimeFlexibleDense layer.

        Parameters:
            time_steps (int): Number of time steps in the sequence data.
            input_size (int): Number of input features per time step.
            output_size (int): Number of neurons in the dense layer.
            use_bias (bool): Whether to include a bias term in the computation. Default is True.
            batch_size (int): Number of samples per batch. Default is 32.
            activation (str): Activation function to use (e.g., 'leaky_relu', 'selu', 'elu', 'sigmoid', 'tanh').
            alpha (float, optional): Parameter for activation functions (e.g., slope in 'leaky_relu').
            lambda_ (float, optional): Scaling factor for activations like 'selu' or 'elu'.
            train_weights (bool): Whether to allow training of weights. Default is True.
            train_bias (bool): Whether to allow training of biases. Default is True.
            train_alpha (bool): Whether to allow training of the alpha parameter. Default is True.
            train_lambda (bool): Whether to allow training of the lambda parameter. Default is True.
            weights_init_method (str): Initialization method for weights (e.g., 'he', 'xavier').
            L2_coe (float): Coefficient for L2 regularization. Default is 0.0.
            L1_coe (float): Coefficient for L1 regularization. Default is 0.0.
            weight_distribution (str): Distribution for weight initialization ('normal' or 'uniform').
            orthogonal_scale_factor (float): Scale factor for orthogonal initialization.
            weights_uniform_range (tuple, optional): Range for uniform weight distribution.

        Returns:
            None
        """
        self.output_size = output_size  # Number of output neurons
        self.input_size = input_size  # Number of input features
        self.batch_size = batch_size  # Number of samples per batch
        self.time_steps = time_steps  # Number of time steps in the sequence
        self.L2_coe = L2_coe  # L2 regularization coefficient
        self.L1_coe = L1_coe  # L1 regularization coefficient
        self.use_bias = use_bias  # Whether to use bias in the computation
        self.activation = activation  # Activation function to use
        self.train_weights = train_weights  # Whether weights are trainable
        self.train_alpha = train_alpha  # Whether alpha parameter is trainable
        self.train_bias = train_bias if use_bias else False  # Train bias only if bias is enabled
        self.train_lambda = train_lambda if activation in ['selu', 'elu'] else False  # Train lambda only for SELU or ELU

        # Initialize weight matrix
        self.weight = Dense_weight_init(
            input_size, output_size, method=weights_init_method,
            distribution=weight_distribution, scale_factor=orthogonal_scale_factor,
            ranges=weights_uniform_range
        )

        # Initialize bias vector if enabled
        if self.use_bias:
            self.bias = np.zeros((output_size, 1))  # Bias initialized to zero

        # Initialize alpha and lambda for specific activations
        if alpha is None:
            alpha = 0.01 if self.activation == 'leaky_relu' else 1.0
        self.alpha = alpha + np.zeros((output_size, 1))  # Shape alpha as a vector
        self.lambda_param = None
        if self.activation in ['selu', 'elu']:
            self.lambda_param = (lambda_ if lambda_ is not None else 1.0) + np.zeros((output_size, 1))

        # Initialize gradients for trainable parameters
        self.grad_w = np.zeros(self.weight.shape) if self.train_weights else None
        self.grad_b = np.zeros(self.bias.shape) if self.train_bias else None
        self.grad_alpha = np.zeros(self.alpha.shape) if self.train_alpha else None
        self.grad_lambda = np.zeros(self.lambda_param.shape) if self.train_lambda else None

        # Initialize placeholders for input, net (pre-activation), and output (post-activation) values
        self.input = np.zeros((batch_size, time_steps, input_size, 1))  # Placeholder for input
        self.net = np.zeros((batch_size, time_steps, output_size, 1))  # Placeholder for pre-activation values
        self.output = np.zeros((batch_size, time_steps, output_size, 1))  # Placeholder for activated outputs

    #################################################################

    def trainable_params(self) -> int:
        """
        Computes the total number of trainable parameters in the layer.

        Returns:
            int: Total number of trainable weights, biases, alpha, and lambda parameters.
        """
        params = 0
        if self.train_weights:
            params += np.size(self.weight)  # Count trainable weights
        if self.train_bias:
            params += np.size(self.bias)  # Count trainable biases
        if self.train_alpha:
            params += np.size(self.alpha)  # Count trainable alpha parameters
        if self.train_lambda:
            params += np.size(self.lambda_param)  # Count trainable lambda parameters
        return params
    
    #################################################################

    def all_params(self) -> int:
        """
        Computes the total number of parameters in the layer (trainable and non-trainable).

        Returns:
            int: Total number of all weights, biases, alpha, and lambda parameters.
        """
        params = np.size(self.alpha) + np.size(self.weight)  # Include weights and alpha
        if self.use_bias:
            params += np.size(self.bias)  # Include biases if enabled
        if self.lambda_param is not None:
            params += np.size(self.lambda_param)  # Include lambda if applicable
        return params

    #################################################################

    def __call__(self, batch_index: int, seq_index: int, input: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the layer.

        Parameters:
            batch_index (int): Index of the current batch being processed.
            seq_index (int): Index of the current time step within the sequence.
            input (np.ndarray): Input data array of shape (batch_size, time_steps, input_size, 1).

        Returns:
            np.ndarray: Output data array of shape (batch_size, output_size, 1).
        """
        # Store the input for this batch and sequence
        self.input[batch_index, seq_index] = input.reshape((-1, 1))

        # Validate that the batch size is within the allowed range
        if self.batch_size < input.shape[0]:
            raise ValueError('Data batch size cannot be larger than model batch size')

        # Perform the linear transformation: net = Wx + b
        self.net[batch_index, seq_index] = self.weight @ input.reshape((-1, 1))
        if self.use_bias:
            self.net[batch_index, seq_index] += self.bias

        # Apply the activation function
        if self.activation in ['selu', 'elu']:
            self.output[batch_index, seq_index] = net2out(
                self.net[batch_index, seq_index], self.activation, alpha=self.alpha, lambda_param=self.lambda_param
            )
        else:
            self.output[batch_index, seq_index] = net2out(
                self.net[batch_index, seq_index], self.activation, alpha=self.alpha
            )

        return self.output[batch_index, seq_index]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for the layer using the specified method.

        Parameters:
            optimizer (str): The optimization algorithm to use (default is 'Adam').
            **kwargs: Additional parameters for configuring the optimizer.

        Returns:
            None
        """
        # Set up the optimizer for trainable parameters in the layer
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, batch_size: int, learning_rate: float = 1e-3, grads: np.ndarray = None) -> None:
        """
        Updates the layer's weights, biases, alpha, and lambda parameters based on calculated gradients.

        Parameters:
            batch_size (int): Number of samples in the current batch.
            learning_rate (float): Learning rate for parameter updates. Defaults to 1e-3.
            grads (np.ndarray, optional): Gradients of the parameters. If None, uses internally computed gradients.

        Returns:
            None
        """
        # Apply L1 and L2 regularization to weights if trainable
        if grads is None:
            if self.train_weights:
                self.grad_w += (self.L1_coe * np.sign(self.weight) + self.L2_coe * self.weight)

            # Combine all gradients into a single array
            grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
            if grads is not None:
                if self.grad_w is not None:
                    grads = np.concatenate((grads, self.grad_w.reshape((-1, 1))))
                if self.grad_b is not None:
                    grads = np.concatenate((grads, self.grad_b.reshape((-1, 1))))
                if self.grad_alpha is not None:
                    grads = np.concatenate((grads, self.grad_alpha.reshape((-1, 1))))
                if self.grad_lambda is not None:
                    grads = np.concatenate((grads, self.grad_lambda.reshape((-1, 1))))
                grads /= batch_size  # Normalize gradients by batch size

        # Use the optimizer to compute parameter updates (deltas)
        deltas = self.Optimizer(grads, learning_rate)

        # Apply updates to weights, biases, alpha, and lambda parameters
        ind2 = 0
        if self.train_weights:
            ind1 = ind2
            ind2 += int(np.size(self.weight))
            delta_w = deltas[ind1:ind2].reshape(self.weight.shape)
            self.weight -= delta_w

        if self.train_bias:
            ind1 = ind2
            ind2 += np.size(self.bias)
            delta_bias = deltas[ind1:ind2].reshape(self.bias.shape)
            self.bias -= delta_bias

        if self.train_alpha:
            ind1 = ind2
            ind2 += np.size(self.alpha)
            delta_alpha = deltas[ind1:ind2].reshape(self.alpha.shape)
            self.alpha -= delta_alpha

        if self.train_lambda:
            ind1 = ind2
            ind2 += np.size(self.lambda_param)
            delta_lambda = deltas[ind1:ind2].reshape(self.lambda_param.shape)
            self.lambda_param -= delta_lambda

        # Reset gradients for the next iteration
        self.grad_w = self.grad_w * 0 if self.train_weights else None
        self.grad_b = self.grad_b * 0 if self.train_bias else None
        self.grad_alpha = self.grad_alpha * 0 if self.train_alpha else None
        self.grad_lambda = self.grad_lambda * 0 if self.train_lambda else None

    #################################################################

    def return_grads(self) -> np.ndarray:
        """
        Returns the accumulated gradients for all trainable parameters.

        Returns:
            np.ndarray: Array containing gradients for weights, biases, alpha, and lambda parameters.
        """
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
        if self.grad_w is not None:
            grad_w = self.grad_w + (self.L1_coe * np.sign(self.weight) + self.L2_coe * self.weight)
            grads = np.concatenate((grads, grad_w.reshape((-1, 1))))
        if self.grad_b is not None:
            grads = np.concatenate((grads, self.grad_b.reshape((-1, 1))))
        if self.grad_alpha is not None:
            grads = np.concatenate((grads, self.grad_alpha.reshape((-1, 1))))
        if self.grad_lambda is not None:
            grads = np.concatenate((grads, self.grad_lambda.reshape((-1, 1))))
        return grads

    #################################################################

    def backward(self, batch_index: int, seq_index: int, error: np.ndarray) -> np.ndarray:
        """
        Performs backpropagation for a single batch and time step, updating gradients and propagating error.

        Parameters:
            batch_index (int): Index of the current batch being processed.
            seq_index (int): Index of the current time step within the sequence.
            error (np.ndarray): Error from the subsequent layer or time step.

        Returns:
            np.ndarray: Propagated error to the previous layer or input.
        """
        # Compute activation derivative and sensitivity
        if self.train_alpha or self.train_lambda:
            Fstar = net2Fstar(self.net[batch_index, seq_index], self.activation, self.alpha, lambda_param=self.lambda_param)
            if self.activation in ['selu', 'elu']:
                if self.train_alpha:
                    self.grad_alpha += error.reshape((-1, 1)) * Fstar[0]
                if self.train_lambda:
                    self.grad_lambda += error.reshape((-1, 1)) * Fstar[1]
            else:
                if self.train_alpha:
                    self.grad_alpha += error.reshape((-1, 1)) * Fstar

        # Compute derivative of the activation function
        Fprime = net2Fprime(self.net[batch_index, seq_index], self.activation, self.alpha, lambda_param=self.lambda_param)

        # Compute sensitivity as error scaled by activation derivative
        sensitivity = error.reshape((-1, 1)) * Fprime

        # Accumulate gradients for weights and biases
        if self.train_weights:
            self.grad_w += np.outer(sensitivity.ravel(), self.input[batch_index, seq_index].ravel())
        if self.train_bias:
            self.grad_b += sensitivity

        # Propagate the error to the previous layer
        error_in = np.ravel(self.weight.T @ sensitivity)

        return error_in.reshape((-1, 1))

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class FlexibleDense1Feedback:
    """
    A flexible Dense layer implementation that supports feedback from the previous state,
    batch processing, and various activation functions.

    Parameters:
        time_steps (int): Number of time steps or sequences.
        input_size (int): Number of input features.
        output_size (int): Number of neurons in the layer.
        feedback_size (int): Size of the feedback (state) vector.
        use_bias (bool): Whether to include a bias term. Default is True.
        batch_size (int): Number of samples per batch. Default is 32.
        activation (str): Activation function to use ('leaky_relu', 'selu', 'elu', 'sigmoid', 'tanh').
        alpha (float, optional): Parameter for activation functions that require an alpha value.
        lambda_ (float, optional): Scaling factor, required for 'selu' and 'elu'.
        train_weights (bool): Whether to allow training of weights. Default is True.
        train_bias (bool): Whether to allow training of biases. Default is True.
        train_alpha (bool): Whether to allow training of the alpha parameter. Default is True.
        train_lambda (bool): Whether to allow training of the lambda parameter. Default is True.
        weights_init_method (str): Weight initialization method (e.g., 'he', 'xavier').
        L2_coe (float, optional): L2 regularization coefficient. Default is 0.0.
        L1_coe (float, optional): L1 regularization coefficient. Default is 0.0.
        weight_distribution (str): Distribution of weights ('normal' or 'uniform').
        orthogonal_scale_factor (float): Scale factor for orthogonal initialization.
        weights_uniform_range (tuple, optional): Range for uniform weight distribution.

    Attributes:
        weight (np.ndarray): Weight matrix of shape (output_size, input_size).
        weight_state (np.ndarray): Weight matrix for the feedback state.
        bias (np.ndarray): Bias vector of shape (output_size, 1), initialized to zero if `use_bias` is True.
        alpha (np.ndarray): Alpha parameter for activation, shaped (output_size, 1).
        lambda_param (np.ndarray): Lambda parameter for SELU or ELU activation, if applicable.
        net (np.ndarray): Pre-activation values for each neuron in each batch.
        output (np.ndarray): Activated output values for each neuron in each batch.
    """

    def __init__(self, time_steps: int, input_size: int, output_size: int, use_bias: bool = True, batch_size: int = 32, 
                 activation: str = 'leaky_relu', alpha: float = None, lambda_=None, feedback_size: int = None,
                 train_weights: bool = True, train_bias: bool = True, train_alpha: bool = True, train_lambda: bool = True,
                 weights_init_method: str = 'he', L2_coe: float = 0.0, L1_coe: float = 0.0,
                 weight_distribution: str = 'normal', orthogonal_scale_factor: float = 1.0, 
                 weights_uniform_range: tuple = None):
        """
        Initializes the FlexibleDense1Feedback layer.

        Parameters:
            time_steps (int): Number of time steps or sequences in the input data.
            input_size (int): Number of input features per time step.
            output_size (int): Number of neurons in the dense layer.
            use_bias (bool): Whether to include a bias term in the computation. Default is True.
            batch_size (int): Number of samples per batch. Default is 32.
            activation (str): Activation function to use ('leaky_relu', 'selu', 'elu', 'sigmoid', 'tanh').
            alpha (float, optional): Parameter for activation functions (e.g., slope in 'leaky_relu').
            lambda_ (float, optional): Scaling factor for activations like 'selu' or 'elu'.
            feedback_size (int, optional): Size of the feedback (state) vector. Defaults to output_size.
            train_weights (bool): Whether to allow training of weights. Default is True.
            train_bias (bool): Whether to allow training of biases. Default is True.
            train_alpha (bool): Whether to allow training of the alpha parameter. Default is True.
            train_lambda (bool): Whether to allow training of the lambda parameter. Default is True.
            weights_init_method (str): Initialization method for weights (e.g., 'he', 'xavier').
            L2_coe (float): Coefficient for L2 regularization. Default is 0.0.
            L1_coe (float): Coefficient for L1 regularization. Default is 0.0.
            weight_distribution (str): Distribution for weight initialization ('normal' or 'uniform').
            orthogonal_scale_factor (float): Scale factor for orthogonal initialization.
            weights_uniform_range (tuple, optional): Range for uniform weight distribution.

        Returns:
            None
        """
        # Set layer configuration
        self.output_size = output_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.feedback_size = feedback_size if feedback_size is not None else output_size
        self.time_steps = time_steps
        self.L2_coe = L2_coe
        self.L1_coe = L1_coe
        self.use_bias = use_bias
        self.activation = activation
        self.train_weights = train_weights
        self.train_alpha = train_alpha
        self.train_bias = train_bias if use_bias else False
        self.train_lambda = train_lambda if activation in ['selu', 'elu'] else False

        # Initialize input-to-output weights
        self.weight = Dense_weight_init(
            input_size, output_size, method=weights_init_method,
            distribution=weight_distribution, scale_factor=orthogonal_scale_factor,
            ranges=weights_uniform_range
        )

        # Initialize feedback-to-output weights
        self.weight_state = Dense_weight_init(
            self.feedback_size, output_size, method=weights_init_method,
            distribution=weight_distribution, scale_factor=orthogonal_scale_factor,
            ranges=weights_uniform_range
        )

        # Initialize bias vector if applicable
        if self.use_bias:
            self.bias = np.zeros((output_size, 1))  # Bias initialized to zero

        # Initialize alpha parameter
        if alpha is None:
            alpha = 0.01 if self.activation == 'leaky_relu' else 1.0
        self.alpha = alpha + np.zeros((output_size, 1))  # Ensure alpha has correct shape

        # Initialize lambda parameter for SELU or ELU activation
        self.lambda_param = None
        if self.activation in ['selu', 'elu']:
            self.lambda_param = (lambda_ if lambda_ is not None else 1.0) + np.zeros((output_size, 1))

        # Initialize gradients for trainable parameters
        self.grad_w = np.zeros(self.weight.shape) if self.train_weights else None
        self.grad_w_state = np.zeros(self.weight_state.shape) if self.train_weights else None
        self.grad_b = np.zeros(self.bias.shape) if self.train_bias else None
        self.grad_alpha = np.zeros(self.alpha.shape) if self.train_alpha else None
        self.grad_lambda = np.zeros(self.lambda_param.shape) if self.train_lambda else None

        # Allocate memory for input, net (pre-activation), and output (post-activation) values
        self.input = np.zeros((batch_size, time_steps, input_size, 1))
        self.net = np.zeros((batch_size, time_steps, output_size, 1))
        self.output = np.zeros((batch_size, time_steps, output_size, 1))

    #################################################################

    def trainable_params(self) -> int:
        """
        Computes the total number of trainable parameters in the layer.

        Returns:
            int: Total number of trainable weights, biases, alpha, and lambda parameters.
        """
        params = 0
        if self.train_weights:
            params += np.size(self.weight)  # Add weights
            params += np.size(self.weight_state)  # Add feedback weights
        if self.train_bias:
            params += np.size(self.bias)  # Add biases
        if self.train_alpha:
            params += np.size(self.alpha)  # Add alpha parameters
        if self.train_lambda:
            params += np.size(self.lambda_param)  # Add lambda parameters
        return params
    
    #################################################################

    def all_params(self) -> int:
        """
        Computes the total number of parameters in the layer (trainable and non-trainable).

        Returns:
            int: Total number of weights, biases, alpha, and lambda parameters.
        """
        params = np.size(self.weight) + np.size(self.weight_state)  # Add weights and feedback weights
        params += np.size(self.alpha)  # Add alpha parameters
        if self.use_bias:
            params += np.size(self.bias)  # Add biases if applicable
        if self.lambda_param is not None:
            params += np.size(self.lambda_param)  # Add lambda parameters if applicable
        return params

    #################################################################

    def __call__(self, batch_index: int, seq_index: int, input: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the layer.

        Parameters:
            batch_index (int): Index of the current batch being processed.
            seq_index (int): Index of the current time step within the sequence.
            input (np.ndarray): Input data array of shape (batch_size, time_steps, input_size, 1).
            state (np.ndarray): Feedback state array of shape (feedback_size, 1).

        Returns:
            np.ndarray: Activated output of the layer for the current time step, shape (output_size, 1).
        """
        # Store the input for the current batch and sequence
        self.input[batch_index, seq_index] = input.reshape((-1, 1))

        # Validate batch size
        if self.batch_size < input.shape[0]:
            raise ValueError('Data batch size cannot exceed the model batch size.')

        # Compute the linear transformation: net = Wx + W_state * state + b
        self.net[batch_index, seq_index] = self.weight @ input.reshape((-1, 1)) + \
                                           self.weight_state @ state.reshape((-1, 1))
        if self.use_bias:
            self.net[batch_index, seq_index] += self.bias

        # Apply the activation function
        if self.activation in ['selu', 'elu']:
            self.output[batch_index, seq_index] = net2out(
                self.net[batch_index, seq_index], self.activation, alpha=self.alpha, lambda_param=self.lambda_param
            )
        else:
            self.output[batch_index, seq_index] = net2out(
                self.net[batch_index, seq_index], self.activation, alpha=self.alpha
            )

        return self.output[batch_index, seq_index]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for the layer's trainable parameters.

        Parameters:
            optimizer (str): Name of the optimizer (default is 'Adam').
            **kwargs: Additional parameters for configuring the optimizer.

        Returns:
            None
        """
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, batch_size: int, learning_rate: float = 1e-3, grads: np.ndarray = None) -> None:
        """
        Updates the layer's weights, biases, alpha, and lambda parameters based on calculated gradients.

        Parameters:
            batch_size (int): Number of samples in the current batch.
            learning_rate (float): Learning rate for parameter updates. Default is 1e-3.
            grads (np.ndarray, optional): Gradients for the parameters. If None, uses internally computed gradients.

        Returns:
            None
        """
        if grads is None:
            # Apply L1 and L2 regularization if weights are trainable
            if self.train_weights:
                self.grad_w += (self.L1_coe * np.sign(self.weight) + self.L2_coe * self.weight)
                self.grad_w_state += (self.L1_coe * np.sign(self.weight_state) + self.L2_coe * self.weight_state)

            # Combine gradients into a single array
            grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
            if grads is not None:
                if self.grad_w is not None:
                    grads = np.concatenate((grads, self.grad_w.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_state.reshape((-1, 1))))
                if self.grad_b is not None:
                    grads = np.concatenate((grads, self.grad_b.reshape((-1, 1))))
                if self.grad_alpha is not None:
                    grads = np.concatenate((grads, self.grad_alpha.reshape((-1, 1))))
                if self.grad_lambda is not None:
                    grads = np.concatenate((grads, self.grad_lambda.reshape((-1, 1))))
                grads /= batch_size

        # Compute parameter updates using the optimizer
        deltas = self.Optimizer(grads, learning_rate)

        # Apply updates to parameters
        ind2 = 0
        if self.train_weights:
            ind1 = ind2
            ind2 += np.size(self.weight)
            self.weight -= deltas[ind1:ind2].reshape(self.weight.shape)

            ind1 = ind2
            ind2 += np.size(self.weight_state)
            self.weight_state -= deltas[ind1:ind2].reshape(self.weight_state.shape)

        if self.train_bias:
            ind1 = ind2
            ind2 += np.size(self.bias)
            self.bias -= deltas[ind1:ind2].reshape(self.bias.shape)

        if self.train_alpha:
            ind1 = ind2
            ind2 += np.size(self.alpha)
            self.alpha -= deltas[ind1:ind2].reshape(self.alpha.shape)

        if self.train_lambda:
            ind1 = ind2
            ind2 += np.size(self.lambda_param)
            self.lambda_param -= deltas[ind1:ind2].reshape(self.lambda_param.shape)

        # Reset gradients for the next batch
        self.grad_w = self.grad_w * 0 if self.train_weights else None
        self.grad_w_state = self.grad_w_state * 0 if self.train_weights else None
        self.grad_b = self.grad_b * 0 if self.train_bias else None
        self.grad_alpha = self.grad_alpha * 0 if self.train_alpha else None
        self.grad_lambda = self.grad_lambda * 0 if self.train_lambda else None

    #################################################################

    def return_grads(self) -> np.ndarray:
        """
        Returns the accumulated gradients for all trainable parameters.

        Returns:
            np.ndarray: A single array containing all gradients.
        """
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
        if self.grad_w is not None:
            grads = np.concatenate((grads, self.grad_w.reshape((-1, 1))))
            grads = np.concatenate((grads, self.grad_w_state.reshape((-1, 1))))
        if self.grad_b is not None:
            grads = np.concatenate((grads, self.grad_b.reshape((-1, 1))))
        if self.grad_alpha is not None:
            grads = np.concatenate((grads, self.grad_alpha.reshape((-1, 1))))
        if self.grad_lambda is not None:
            grads = np.concatenate((grads, self.grad_lambda.reshape((-1, 1))))
        return grads

    #################################################################

    def backward(self, batch_index: int, seq_index: int, error: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs backpropagation for a single time step, updating gradients and propagating error.

        Parameters:
            batch_index (int): Index of the current batch being processed.
            seq_index (int): Index of the current time step within the sequence.
            error (np.ndarray): Error from the subsequent layer or time step.
            state (np.ndarray): Feedback state from the current layer.

        Returns:
            tuple[np.ndarray, np.ndarray]: Errors propagated to the input and feedback state.
        """
        # Compute activation derivative
        Fprime = net2Fprime(self.net[batch_index, seq_index], self.activation, self.alpha, lambda_param=self.lambda_param)

        # Compute sensitivity as the product of error and activation derivative
        sensitivity = error.reshape((-1, 1)) * Fprime

        # Accumulate gradients for weights, biases, and feedback weights
        if self.train_weights:
            self.grad_w += np.outer(sensitivity.ravel(), self.input[batch_index, seq_index].ravel())
            self.grad_w_state += np.outer(sensitivity.ravel(), state.ravel())
        if self.train_bias:
            self.grad_b += sensitivity

        # Compute errors to propagate
        error_in = np.ravel(self.weight.T @ sensitivity)
        error_state = np.ravel(self.weight_state.T @ sensitivity)

        return error_in.reshape((-1, 1)), error_state.reshape((-1, 1))
    
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class FlexibleDense2Feedback:
    """
    A flexible Dense layer implementation with two feedback mechanisms (Elman and Jordan),
    supporting various activation functions, initialization options, and batch processing.

    Parameters:
        time_steps (int): Number of time steps or sequences.
        input_size (int): Number of input features.
        output_size (int): Number of neurons in the layer.
        feedback_size_jordan (int): Number of rows in the Jordan feedback state vector.
        use_bias (bool): Whether to include a bias term. Default is True.
        batch_size (int): Number of samples per batch. Default is 32.
        activation (str): Activation function to use ('leaky_relu', 'selu', 'elu', 'sigmoid', 'tanh').
        alpha (float, optional): Parameter for activation functions that require an alpha value.
        lambda_ (float, optional): Scaling factor, required for 'selu' and 'elu'.
        train_weights (bool): Whether to allow training of weights. Default is True.
        train_bias (bool): Whether to allow training of biases. Default is True.
        train_alpha (bool): Whether to allow training of the alpha parameter. Default is True.
        train_lambda (bool): Whether to allow training of the lambda parameter. Default is True.
        weights_init_method (str): Initialization method for weights (e.g., 'he', 'xavier').
        L2_coe (float, optional): L2 regularization coefficient. Default is 0.0.
        L1_coe (float, optional): L1 regularization coefficient. Default is 0.0.
        weight_distribution (str): Distribution for weight initialization ('normal' or 'uniform').
        orthogonal_scale_factor (float): Scale factor for orthogonal initialization.
        weights_uniform_range (tuple, optional): Range for uniform weight distribution.

    Attributes:
        weight (np.ndarray): Weight matrix of shape (output_size, input_size).
        weight_elman (np.ndarray): Weight matrix for Elman feedback.
        weight_jordan (np.ndarray): Weight matrix for Jordan feedback.
        bias (np.ndarray): Bias vector of shape (output_size, 1), initialized to zero if `use_bias` is True.
        alpha (np.ndarray): Alpha parameter for activation, shaped (output_size, 1).
        lambda_param (np.ndarray): Lambda parameter for SELU or ELU activation, if applicable.
        net (np.ndarray): Pre-activation values for each neuron in each batch.
        output (np.ndarray): Activated output values for each neuron in each batch.
    """

    def __init__(self, time_steps: int, input_size: int, output_size: int, feedback_size_jordan: int,
                 use_bias: bool = True, batch_size: int = 32, activation: str = 'leaky_relu', alpha: float = None,
                 lambda_=None, train_weights: bool = True, train_bias: bool = True, train_alpha: bool = True,
                 train_lambda: bool = True, weights_init_method: str = 'he', L2_coe: float = 0.0,
                 L1_coe: float = 0.0, weight_distribution: str = 'normal', orthogonal_scale_factor: float = 1.0,
                 weights_uniform_range: tuple = None):
        """
        Initializes the FlexibleDense2Feedback layer.

        Parameters:
            time_steps (int): Number of time steps or sequences in the input data.
            input_size (int): Number of input features per time step.
            output_size (int): Number of neurons in the dense layer.
            feedback_size_jordan (int): Number of rows in the Jordan feedback state vector.
            use_bias (bool): Whether to include a bias term. Default is True.
            batch_size (int): Number of samples per batch. Default is 32.
            activation (str): Activation function to use ('leaky_relu', 'selu', 'elu', 'sigmoid', 'tanh').
            alpha (float, optional): Parameter for activation functions (e.g., slope in 'leaky_relu').
            lambda_ (float, optional): Scaling factor for activations like 'selu' or 'elu'.
            train_weights (bool): Whether to allow training of weights. Default is True.
            train_bias (bool): Whether to allow training of biases. Default is True.
            train_alpha (bool): Whether to allow training of the alpha parameter. Default is True.
            train_lambda (bool): Whether to allow training of the lambda parameter. Default is True.
            weights_init_method (str): Initialization method for weights (e.g., 'he', 'xavier').
            L2_coe (float): Coefficient for L2 regularization. Default is 0.0.
            L1_coe (float): Coefficient for L1 regularization. Default is 0.0.
            weight_distribution (str): Distribution for weight initialization ('normal' or 'uniform').
            orthogonal_scale_factor (float): Scale factor for orthogonal initialization.
            weights_uniform_range (tuple, optional): Range for uniform weight distribution.

        Returns:
            None
        """
        # Layer configuration
        self.output_size = output_size
        self.input_size = input_size
        self.feedback_size_elman = output_size  # Feedback from the current layer
        self.feedback_size_jordan = feedback_size_jordan  # Feedback from the previous layer
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.use_bias = use_bias
        self.activation = activation
        self.train_weights = train_weights
        self.train_bias = train_bias if use_bias else False
        self.train_alpha = train_alpha
        self.train_lambda = train_lambda if activation in ['selu', 'elu'] else False
        self.L2_coe = L2_coe
        self.L1_coe = L1_coe

        # Initialize weights
        self.weight = Dense_weight_init(input_size, output_size, method=weights_init_method,
                                        distribution=weight_distribution, scale_factor=orthogonal_scale_factor,
                                        ranges=weights_uniform_range)
        self.weight_elman = Dense_weight_init(output_size, output_size, method=weights_init_method,
                                              distribution=weight_distribution, scale_factor=orthogonal_scale_factor,
                                              ranges=weights_uniform_range)
        self.weight_jordan = Dense_weight_init(feedback_size_jordan, output_size, method=weights_init_method,
                                               distribution=weight_distribution, scale_factor=orthogonal_scale_factor,
                                               ranges=weights_uniform_range)

        # Initialize biases
        if self.use_bias:
            self.bias = np.zeros((output_size, 1))  # Bias initialized to zero

        # Initialize alpha and lambda parameters
        self.alpha = np.full((output_size, 1), 0.01 if alpha is None else alpha)
        self.lambda_param = np.full((output_size, 1), 1.0 if lambda_ is None else lambda_) if activation in ['selu', 'elu'] else None

        # Initialize gradients
        self.grad_w = np.zeros(self.weight.shape) if self.train_weights else None
        self.grad_w_elman = np.zeros(self.weight_elman.shape) if self.train_weights else None
        self.grad_w_jordan = np.zeros(self.weight_jordan.shape) if self.train_weights else None
        self.grad_b = np.zeros(self.bias.shape) if self.train_bias else None
        self.grad_alpha = np.zeros(self.alpha.shape) if self.train_alpha else None
        self.grad_lambda = np.zeros(self.lambda_param.shape) if self.train_lambda else None

        # Allocate memory for input, net, and output
        self.input = np.zeros((batch_size, time_steps, input_size, 1))
        self.net = np.zeros((batch_size, time_steps, output_size, 1))
        self.output = np.zeros((batch_size, time_steps, output_size, 1))

    #################################################################

    def trainable_params(self) -> int:
        """
        Computes the total number of trainable parameters in the layer.

        Returns:
            int: Total number of trainable weights, biases, alpha, and lambda parameters.
        """
        params = 0
        if self.train_weights:
            params += np.size(self.weight)  # Add input-to-output weights
            params += np.size(self.weight_elman)  # Add Elman feedback weights
            params += np.size(self.weight_jordan)  # Add Jordan feedback weights
        if self.train_bias:
            params += np.size(self.bias)  # Add biases
        if self.train_alpha:
            params += np.size(self.alpha)  # Add alpha parameters
        if self.train_lambda:
            params += np.size(self.lambda_param)  # Add lambda parameters
        return params
    
    #################################################################

    def all_params(self) -> int:
        """
        Computes the total number of parameters in the layer (trainable and non-trainable).

        Returns:
            int: Total number of weights, biases, alpha, and lambda parameters.
        """
        params = np.size(self.weight) + np.size(self.weight_elman) + np.size(self.weight_jordan)
        params += np.size(self.alpha)  # Add alpha parameters
        if self.use_bias:
            params += np.size(self.bias)  # Add biases if applicable
        if self.lambda_param is not None:
            params += np.size(self.lambda_param)  # Add lambda parameters if applicable
        return params

    #################################################################

    def __call__(self, batch_index: int, seq_index: int, input: np.ndarray, 
                 elman_state: np.ndarray, jordan_state: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the layer.

        Parameters:
            batch_index (int): Index of the current batch being processed.
            seq_index (int): Index of the current time step within the sequence.
            input (np.ndarray): Input data array of shape (input_size, 1).
            elman_state (np.ndarray): Feedback state from the current layer, shape (output_size, 1).
            jordan_state (np.ndarray): Feedback state from the previous layer, shape (feedback_size_jordan, 1).

        Returns:
            np.ndarray: Activated output of the layer for the current time step, shape (output_size, 1).
        """
        # Store the input for the current batch and sequence
        self.input[batch_index, seq_index] = input.reshape((-1, 1))

        # Validate batch size
        if self.batch_size < input.shape[0]:
            raise ValueError('Data batch size cannot exceed the model batch size.')

        # Compute the linear transformation: net = Wx + W_elman * elman_state + W_jordan * jordan_state + b
        self.net[batch_index, seq_index] = self.weight @ input + \
                                           self.weight_elman @ elman_state + \
                                           self.weight_jordan @ jordan_state
        if self.use_bias:
            self.net[batch_index, seq_index] += self.bias

        # Apply the activation function
        if self.activation in ['selu', 'elu']:
            self.output[batch_index, seq_index] = net2out(
                self.net[batch_index, seq_index], self.activation, alpha=self.alpha, lambda_param=self.lambda_param
            )
        else:
            self.output[batch_index, seq_index] = net2out(
                self.net[batch_index, seq_index], self.activation, alpha=self.alpha
            )

        return self.output[batch_index, seq_index]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for the layer's trainable parameters.

        Parameters:
            optimizer (str): Name of the optimizer (default is 'Adam').
            **kwargs: Additional parameters for configuring the optimizer.

        Returns:
            None
        """
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, batch_size: int, learning_rate: float = 1e-3, grads: np.ndarray = None) -> None:
        """
        Updates the layer's weights, biases, alpha, and lambda parameters based on calculated gradients.

        Parameters:
            batch_size (int): Number of samples in the current batch.
            learning_rate (float): Learning rate for parameter updates. Default is 1e-3.
            grads (np.ndarray, optional): Gradients for the parameters. If None, uses internally computed gradients.

        Returns:
            None
        """
        if grads is None:
            # Apply L1 and L2 regularization to weights if trainable
            if self.train_weights:
                self.grad_w += (self.L1_coe * np.sign(self.weight) + self.L2_coe * self.weight)
                self.grad_w_elman += (self.L1_coe * np.sign(self.weight_elman) + self.L2_coe * self.weight_elman)
                self.grad_w_jordan += (self.L1_coe * np.sign(self.weight_jordan) + self.L2_coe * self.weight_jordan)

            # Combine gradients into a single array
            grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
            if grads is not None:
                if self.grad_w is not None:
                    grads = np.concatenate((grads, self.grad_w.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_elman.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_jordan.reshape((-1, 1))))
                if self.grad_b is not None:
                    grads = np.concatenate((grads, self.grad_b.reshape((-1, 1))))
                if self.grad_alpha is not None:
                    grads = np.concatenate((grads, self.grad_alpha.reshape((-1, 1))))
                if self.grad_lambda is not None:
                    grads = np.concatenate((grads, self.grad_lambda.reshape((-1, 1))))
                grads /= batch_size

        # Compute parameter updates using the optimizer
        deltas = self.Optimizer(grads, learning_rate)

        # Apply updates to parameters
        ind2 = 0
        if self.train_weights:
            ind1 = ind2
            ind2 += np.size(self.weight)
            self.weight -= deltas[ind1:ind2].reshape(self.weight.shape)

            ind1 = ind2
            ind2 += np.size(self.weight_elman)
            self.weight_elman -= deltas[ind1:ind2].reshape(self.weight_elman.shape)

            ind1 = ind2
            ind2 += np.size(self.weight_jordan)
            self.weight_jordan -= deltas[ind1:ind2].reshape(self.weight_jordan.shape)

        if self.train_bias:
            ind1 = ind2
            ind2 += np.size(self.bias)
            self.bias -= deltas[ind1:ind2].reshape(self.bias.shape)

        if self.train_alpha:
            ind1 = ind2
            ind2 += np.size(self.alpha)
            self.alpha -= deltas[ind1:ind2].reshape(self.alpha.shape)

        if self.train_lambda:
            ind1 = ind2
            ind2 += np.size(self.lambda_param)
            self.lambda_param -= deltas[ind1:ind2].reshape(self.lambda_param.shape)

        # Reset gradients for the next batch
        self.grad_w = self.grad_w * 0 if self.train_weights else None
        self.grad_w_elman = self.grad_w_elman * 0 if self.train_weights else None
        self.grad_w_jordan = self.grad_w_jordan * 0 if self.train_weights else None
        self.grad_b = self.grad_b * 0 if self.train_bias else None
        self.grad_alpha = self.grad_alpha * 0 if self.train_alpha else None
        self.grad_lambda = self.grad_lambda * 0 if self.train_lambda else None

    #################################################################

    def return_grads(self) -> np.ndarray:
        """
        Returns the accumulated gradients for all trainable parameters.

        Returns:
            np.ndarray: A single array containing all gradients.
        """
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
        if self.grad_w is not None:
            grads = np.concatenate((grads, self.grad_w.reshape((-1, 1))))
            grads = np.concatenate((grads, self.grad_w_elman.reshape((-1, 1))))
            grads = np.concatenate((grads, self.grad_w_jordan.reshape((-1, 1))))
        if self.grad_b is not None:
            grads = np.concatenate((grads, self.grad_b.reshape((-1, 1))))
        if self.grad_alpha is not None:
            grads = np.concatenate((grads, self.grad_alpha.reshape((-1, 1))))
        if self.grad_lambda is not None:
            grads = np.concatenate((grads, self.grad_lambda.reshape((-1, 1))))
        return grads

    #################################################################

    def backward(self, batch_index: int, seq_index: int, error: np.ndarray,
                 elman_state: np.ndarray, jordan_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs backpropagation for a single time step, updating gradients and propagating error.

        Parameters:
            batch_index (int): Index of the current batch being processed.
            seq_index (int): Index of the current time step within the sequence.
            error (np.ndarray): Error from the subsequent layer or time step.
            elman_state (np.ndarray): Feedback state from the current layer.
            jordan_state (np.ndarray): Feedback state from the previous layer.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Errors propagated to the input, Elman state, and Jordan state.
        """
        # Compute activation derivative
        Fprime = net2Fprime(self.net[batch_index, seq_index], self.activation, self.alpha, lambda_param=self.lambda_param)

        # Compute sensitivity as the product of error and activation derivative
        sensitivity = error.reshape((-1, 1)) * Fprime

        # Accumulate gradients for weights, biases, and feedback weights
        if self.train_weights:
            self.grad_w += np.outer(sensitivity.ravel(), self.input[batch_index, seq_index].ravel())
            self.grad_w_elman += np.outer(sensitivity.ravel(), elman_state.ravel())
            self.grad_w_jordan += np.outer(sensitivity.ravel(), jordan_state.ravel())
        if self.train_bias:
            self.grad_b += sensitivity

        # Compute errors to propagate
        error_in = np.ravel(self.weight.T @ sensitivity)
        error_elman = np.ravel(self.weight_elman.T @ sensitivity)
        error_jordan = np.ravel(self.weight_jordan.T @ sensitivity)

        return error_in.reshape((-1, 1)), error_elman.reshape((-1, 1)), error_jordan.reshape((-1, 1))
    