import numpy as np
from activations.flexible_activation_functions import net2out, net2Fprime, net2Fstar
from initializers.weight_initializer import Dense_weight_init


class flexible_Dense:
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
    - weights_init_method (str): Weight initialization method (e.g., 'he' or 'xavier').
    - weight_distribution (str): Distribution of weights ('normal' or 'uniform').
    - orthogonal_scale_factor (float): Scale factor for orthogonal initialization.
    - weights_uniform_range (tuple, optional): Range for uniform weight distribution.

    Attributes:
    - weight (np.ndarray): Weight matrix of shape (output_size, input_size).
    - bias (np.ndarray): Bias vector of shape (output_size, 1), initialized to zero if `use_bias` is True.
    - alpha (np.ndarray): Alpha parameter for activation, shaped (output_size, 1).
    - lambda_param (np.ndarray): Lambda parameter for SELU or ELU activation, if applicable.
    - net (np.ndarray): Pre-activation values for each neuron in each batch.
    - output (np.ndarray): Activated output values for each neuron in each batch.
    """

    def __init__(self, input_size: int, output_size: int, use_bias: bool = True, batch_size: int = 32, 
                 activation: str = 'leaky_relu', alpha: float = None, lambda_=None, weights_init_method: str = 'he', 
                 weight_distribution: str = 'normal', orthogonal_scale_factor: float = 1.0, 
                 weights_uniform_range: tuple = None):
        self.output_size = output_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.use_bias = use_bias
        self.activation = activation

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
        params = np.size(self.weight) + np.size(self.alpha)
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
        # Reshape input to match the expected shape
        input = input.reshape((-1, self.input_size))
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
            
            # Activation
            if self.activation in ['selu', 'elu']:
                self.output[batch_index] = net2out(self.net[batch_index], self.activation, alpha=self.alpha, lambda_param=self.lambda_param)
            else:
                self.output[batch_index] = net2out(self.net[batch_index], self.activation, alpha=self.alpha)

        return self.output[:input.shape[0]].reshape((-1, self.output_size))

    #################################################################

    def Adam_init(self):
        """
        Initializes the Adam optimizer's moment variables for each parameter.
        """
        self.weight_mt = np.zeros(self.weight.shape)
        self.weight_vt = np.zeros(self.weight.shape)
        self.t = 0  # Timestep

        self.alpha_mt = np.zeros(self.alpha.shape)
        self.alpha_vt = np.zeros(self.alpha.shape)

        if self.use_bias:
            self.bias_mt = np.zeros(self.bias.shape)
            self.bias_vt = np.zeros(self.bias.shape)

        if self.activation in ['selu', 'elu']:
            self.lambda_mt = np.zeros(self.lambda_param.shape)
            self.lambda_vt = np.zeros(self.lambda_param.shape)

    #################################################################

    def update(self, grad_w: np.ndarray, grad_bias: np.ndarray, grad_alpha: np.ndarray, grad_lambda: np.ndarray, 
               method: str = 'Adam', learning_rate: float = 1e-3, bias_learning_rate: float = 2e-4, 
               adam_beta1: float = 0.9, adam_beta2: float = 0.99):
        """
        Updates the model parameters using either Adam or gradient descent.

        Parameters:
        - grad_w (np.ndarray): Gradient of the weights.
        - grad_bias (np.ndarray): Gradient of the bias.
        - grad_alpha (np.ndarray): Gradient of alpha parameter.
        - grad_lambda (np.ndarray): Gradient of lambda parameter.
        - method (str): Optimization method ('Adam' or 'gradient_descent').
        - learning_rate (float): Learning rate for weights.
        - bias_learning_rate (float): Learning rate for bias and additional parameters.
        - adam_beta1 (float): Beta1 parameter for Adam.
        - adam_beta2 (float): Beta2 parameter for Adam.
        """
        if method == 'Adam':
            # Adam updates for weights
            eps = 1e-7
            self.t += 1

            # Update weight moments and compute weight delta
            self.weight_mt = adam_beta1 * self.weight_mt + (1 - adam_beta1) * grad_w
            self.weight_vt = adam_beta2 * self.weight_vt + (1 - adam_beta2) * np.square(grad_w)
            m_hat_w = self.weight_mt / (1 - adam_beta1 ** self.t)
            v_hat_w = self.weight_vt / (1 - adam_beta2 ** self.t)
            delta_w = learning_rate * m_hat_w / (np.sqrt(v_hat_w) + eps)

            # Update alpha moments and compute alpha delta
            self.alpha_mt = adam_beta1 * self.alpha_mt + (1 - adam_beta1) * grad_alpha
            self.alpha_vt = adam_beta2 * self.alpha_vt + (1 - adam_beta2) * np.square(grad_alpha)
            m_hat_alpha = self.alpha_mt / (1 - adam_beta1 ** self.t)
            v_hat_alpha = self.alpha_vt / (1 - adam_beta2 ** self.t)
            delta_alpha = bias_learning_rate * m_hat_alpha / (np.sqrt(v_hat_alpha) + eps)

            # Update bias moments if bias is used
            if self.use_bias:
                self.bias_mt = adam_beta1 * self.bias_mt + (1 - adam_beta1) * grad_bias
                self.bias_vt = adam_beta2 * self.bias_vt + (1 - adam_beta2) * np.square(grad_bias)
                m_hat_bias = self.bias_mt / (1 - adam_beta1 ** self.t)
                v_hat_bias = self.bias_vt / (1 - adam_beta2 ** self.t)
                delta_bias = bias_learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + eps)

            # Update lambda moments for SELU or ELU
            if self.activation in ['selu', 'elu']:
                self.lambda_mt = adam_beta1 * self.lambda_mt + (1 - adam_beta1) * grad_lambda
                self.lambda_vt = adam_beta2 * self.lambda_vt + (1 - adam_beta2) * np.square(grad_lambda)
                m_hat_lambda = self.lambda_mt / (1 - adam_beta1 ** self.t)
                v_hat_lambda = self.lambda_vt / (1 - adam_beta2 ** self.t)
                delta_lambda = bias_learning_rate * m_hat_lambda / (np.sqrt(v_hat_lambda) + eps)
        else:
            # Gradient descent updates
            delta_w = learning_rate * grad_w
            delta_alpha = bias_learning_rate * grad_alpha
            if self.use_bias:
                delta_bias = bias_learning_rate * grad_bias
            if self.activation in ['selu', 'elu']:
                delta_lambda = bias_learning_rate * grad_lambda

        # Apply updates
        self.weight -= delta_w
        self.alpha -= delta_alpha
        if self.use_bias:
            self.bias -= delta_bias
        if self.activation in ['selu', 'elu']:
            self.lambda_param -= delta_lambda

    #################################################################

    def backward(self, error_batch: np.ndarray, method: str = 'Adam', 
                 learning_rate: float = 1e-3, bias_learning_rate: float = 2e-4, 
                 adam_beta1: float = 0.9, adam_beta2: float = 0.99) -> np.ndarray:
        """
        Computes gradients and propagates error backward.

        Parameters:
        - error_batch (np.ndarray): Batch of errors.
        - method (str): Optimization method ('Adam' or 'gradient_descent').
        - learning_rate (float): Learning rate for weights.
        - bias_learning_rate (float): Learning rate for bias and additional parameters.
        - adam_beta1 (float): Beta1 parameter for Adam.
        - adam_beta2 (float): Beta2 parameter for Adam.

        Returns:
        - np.ndarray: Output error propagated to previous layer.
        """
        error_out = np.zeros(self.input.shape)

        # Initialize gradients
        grad_w = np.zeros(self.weight.shape)
        grad_alpha = np.zeros(self.alpha.shape)
        grad_lambda = None
        grad_bias = None
        if self.use_bias:
            grad_bias = np.zeros(self.bias.shape)
        if self.activation in ['selu', 'elu']:
            grad_lambda = np.zeros(self.lambda_param.shape)

        # Compute gradients for each batch sample
        for batch_index, one_batch_error in enumerate(error_batch):
            # Calculate derivatives
            Fstar = net2Fstar(self.net[batch_index], self.activation, self.alpha, lambda_param=self.lambda_param)
            if self.activation in ['selu', 'elu']:
                grad_alpha += one_batch_error.reshape((-1, 1)) * Fstar[0]
                grad_lambda += one_batch_error.reshape((-1, 1)) * Fstar[1]
            else:
                grad_alpha += one_batch_error.reshape((-1, 1)) * Fstar

            Fprime = net2Fprime(self.net[batch_index], self.activation, self.alpha, lambda_param=self.lambda_param)

            # Compute sensitivity
            sensitivity = one_batch_error.reshape((-1, 1)) * Fprime

            # Compute weight gradient
            grad_w += np.outer(sensitivity.ravel(), self.input[batch_index].ravel())
            if self.use_bias:
                grad_bias += sensitivity

            # Backpropagate error
            error_out[batch_index] = np.ravel(self.weight.T @ sensitivity)

        # Average gradients across batch
        grad_w /= error_batch.shape[0]
        grad_alpha /= error_batch.shape[0]
        if self.use_bias:
            grad_bias /= error_batch.shape[0]
        if self.activation in ['selu', 'elu']:
            grad_lambda /= error_batch.shape[0]

        # Update parameters
        self.update(grad_w, grad_bias, grad_alpha, grad_lambda, method=method, learning_rate=learning_rate,
                    bias_learning_rate=bias_learning_rate, adam_beta1=adam_beta1, adam_beta2=adam_beta2)

        return error_out
