import numpy as np
from activations.activation_functions import net2out, net2Fprime
from initializers.weight_initializer import Dense_weight_init


class Dense:
    def __init__(self, input_size: int, output_size: int, use_bias: bool = True, batch_size: int = 32, 
                 activation: str = 'relu', alpha: float = None, weights_init_method: str = 'he', 
                 weight_distribution: str = 'normal', orthogonal_scale_factor: float = 1.0, 
                 weights_uniform_range: tuple = None):
        """
        Initializes a dense (fully connected) layer.
        
        Parameters:
        input_size (int): Number of input neurons.
        output_size (int): Number of output neurons.
        use_bias (bool): Whether to use bias in the layer. Default is True.
        batch_size (int): Batch size for processing. Default is 32.
        activation (str): Activation function to use. Default is 'relu'.
        alpha (float, optional): Parameter for activation functions like Leaky ReLU and ELU. Default is None.
        weights_init_method (str): Method for weight initialization ('xavier', 'he', 'uniform', 'lecun', 'orthogonal'). Default is 'xavier'.
        weight_distribution (str): Distribution type for weight values ('normal', 'uniform'). Default is 'normal'.
        orthogonal_scale_factor (float): Scaling factor for orthogonal initialization. Default is 1.0.
        weights_uniform_range (tuple, optional): Range (min, max) for uniform weight initialization. Default is None.
        """
        self.output_size = output_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.use_bias = use_bias
        self.activation = activation
        self.alpha_activation = alpha  # Alpha value for Leaky ReLU and ELU activation functions

        # Initialize weights using the specified method
        self.weight = Dense_weight_init(input_size, output_size, method=weights_init_method, 
                                        distribution=weight_distribution, scale_factor=orthogonal_scale_factor, 
                                        ranges=weights_uniform_range)
        
        # Initialize bias if required
        if self.use_bias:
            self.bias = np.zeros((output_size, 1))
        
        # Initialize the net and output arrays
        self.net = np.zeros((batch_size, output_size, 1))
        self.output = np.zeros((batch_size, output_size, 1))

    #################################################################

    def trainable_params(self) -> int:
        """
        Returns the number of trainable parameters in the layer.
        
        Returns:
        int: Total number of trainable parameters (weights and biases).
        """
        if self.use_bias:
            return np.size(self.weight) + np.size(self.bias)
        else:
            return np.size(self.weight)

    #################################################################

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the layer.

        Parameters:
        input (np.ndarray): Input data array of shape (batch_size, input_size, 1, channels_size).

        Returns:
        np.ndarray: Output data array of shape (batch_size, output_size, 1).
        """

        # print(input.shape)
        input = input.reshape((-1, self.input_size))

        self.input = input

        # Check if the batch size of the input is valid
        if self.batch_size < input.shape[0]:
            raise ValueError('Data batch size cannot be larger than model batch size')

        # Process each input vector in the batch
        for batch_index, input_vector in enumerate(input):
            self.net[batch_index] = self.weight @ input_vector.reshape((-1, 1))
            if self.use_bias:
                self.net[batch_index] += self.bias
            self.output[batch_index] = net2out(self.net[batch_index], self.activation, alpha=self.alpha_activation)

        return self.output[:input.shape[0]].reshape((-1, self.output_size))

    #################################################################

    def Adam_init(self):
        """
        Initializes parameters for the Adam optimizer.
        """
        # Initialize first moment vector (mt) and second moment vector (vt) for weights
        self.weight_mt = np.zeros(self.weight.shape)
        self.weight_vt = np.zeros(self.weight.shape)
        self.t = 0  # Time step counter

        if self.use_bias:
            # Initialize first and second moment vectors for bias if bias is used
            self.bias_mt = np.zeros(self.bias.shape)
            self.bias_vt = np.zeros(self.bias.shape)

    #################################################################

    def update(self, grad_w: np.ndarray, grad_bias: np.ndarray, method: str = 'Adam', learning_rate: float = 1e-3,
               bias_learning_rate: float = 2e-4, adam_beta1: float = 0.9, adam_beta2: float = 0.99):
        """
        Updates the weights and biases using the specified optimization method.

        Parameters:
        grad_w (np.ndarray): Gradient of the loss with respect to the weights.
        grad_bias (np.ndarray): Gradient of the loss with respect to the biases.
        method (str): Optimization method ('Adam' or 'SGD'). Default is 'Adam'.
        learning_rate (float): Learning rate for weights. Default is 1e-3.
        bias_learning_rate (float): Learning rate for biases. Default is 2e-4.
        adam_beta1 (float): Exponential decay rate for the first moment estimates. Default is 0.9.
        adam_beta2 (float): Exponential decay rate for the second moment estimates. Default is 0.99.
        """
        if method == 'Adam':
            eps = 1e-7  # Small constant to prevent division by zero

            # Update first moment estimate for weights
            self.weight_mt = adam_beta1 * self.weight_mt + (1 - adam_beta1) * grad_w
            # Update second moment estimate for weights
            self.weight_vt = adam_beta2 * self.weight_vt + (1 - adam_beta2) * np.square(grad_w)
            self.t += 1  # Increment time step

            # Compute bias-corrected first moment estimate
            m_hat_w = self.weight_mt / (1 - adam_beta1 ** self.t)
            # Compute bias-corrected second moment estimate
            v_hat_w = self.weight_vt / (1 - adam_beta2 ** self.t)

            # Compute update for weights
            delta_w = learning_rate * m_hat_w / (np.sqrt(v_hat_w) + eps)

            if self.use_bias:
                # Update first moment estimate for biases
                self.bias_mt = adam_beta1 * self.bias_mt + (1 - adam_beta1) * grad_bias
                # Update second moment estimate for biases
                self.bias_vt = adam_beta2 * self.bias_vt + (1 - adam_beta2) * np.square(grad_bias)

                # Compute bias-corrected first moment estimate for biases
                m_hat_bias = self.bias_mt / (1 - adam_beta1 ** self.t)
                # Compute bias-corrected second moment estimate for biases
                v_hat_bias = self.bias_vt / (1 - adam_beta2 ** self.t)

                # Compute update for biases
                delta_bias = bias_learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + eps)
        else:
            # For SGD, the update is simply the learning rate times the gradient
            delta_w = learning_rate * grad_w
            if self.use_bias:
                delta_bias = bias_learning_rate * grad_bias

        # Apply the updates to weights and biases
        self.weight -= delta_w
        if self.use_bias:
            self.bias -= delta_bias

    #################################################################

    def backward(self, error_batch: np.ndarray, method: str = 'Adam', 
                 learning_rate: float = 1e-3, bias_learning_rate: float = 2e-4, 
                 adam_beta1: float = 0.9, adam_beta2: float = 0.99) -> np.ndarray:
        """
        Performs a backward pass and updates the weights and biases.

        Parameters:
        error_batch (np.ndarray): Error gradients from the next layer.
        method (str): Optimization method ('Adam' or 'SGD'). Default is 'Adam'.
        learning_rate (float): Learning rate for weights. Default is 1e-3.
        bias_learning_rate (float): Learning rate for biases. Default is 2e-4.
        adam_beta1 (float): Exponential decay rate for the first moment estimates. Default is 0.9.
        adam_beta2 (float): Exponential decay rate for the second moment estimates. Default is 0.99.

        Returns:
        np.ndarray: Gradients of the error with respect to the input.
        """

        # Initialize the output error gradients array
        error_out = np.zeros(self.input.shape)

        # Initialize the gradients for weights and biases
        grad_w = np.zeros(self.weight.shape)
        grad_bias = None
        if self.use_bias:
            grad_bias = np.zeros(self.bias.shape)

        # Process each error vector in the batch
        for batch_index, one_batch_error in enumerate(error_batch):
            # Compute the derivative of the activation function
            Fprime = net2Fprime(self.net[batch_index], self.activation, self.alpha_activation)
            if self.activation == 'softmax':
                sensitivity = (Fprime @ one_batch_error).reshape((-1, 1))
            else:
                sensitivity = one_batch_error.reshape((-1, 1)) * Fprime

            # Accumulate the gradient for weights
            grad_w += np.outer(sensitivity.ravel(), self.input[batch_index].ravel())
            if self.use_bias:
                grad_bias += sensitivity

            # Compute the error gradient with respect to the input
            error_out[batch_index] = np.ravel(self.weight.T @ sensitivity)

        # Average the gradients over the batch
        grad_w /= error_batch.shape[0]
        if self.use_bias:
            grad_bias /= error_batch.shape[0]

        # Update the weights and biases using the computed gradients
        self.update(grad_w, grad_bias, method=method, learning_rate=learning_rate,
                bias_learning_rate=bias_learning_rate, adam_beta1=adam_beta1, adam_beta2=adam_beta2)

        return error_out
