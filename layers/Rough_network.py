import numpy as np
from activations.activation_functions import net2out, net2Fprime
from initializers.weight_initializer import Dense_weight_init


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
    activation : str, optional
        Activation function to use (default is 'sigmoid'). Other options include 'relu', 'tanh', etc.
    alpha_acti : float, optional
        A scaling parameter for the activation function (default is None).
    weights_uniform_range : tuple, optional
        The range for initializing weights uniformly (default is (-1, 1)).
    train_alpha : bool, optional
        If True, alpha parameter will be trained (default is False).
    """

    def __init__(self, input_size: int, output_size: int, use_bias: bool = True, batch_size: int = 32, 
                 activation: str = 'sigmoid', alpha_acti: float = None, weights_uniform_range: tuple = (-1, 1),
                 train_alpha: bool = False):

        self.output_size = output_size  # Number of output neurons
        self.input_size = input_size  # Number of input neurons/features
        self.batch_size = batch_size  # Batch size
        self.use_bias = use_bias  # Whether to use bias
        self.activation = activation  # Activation function to use
        self.alpha_activation = alpha_acti  # Alpha activation scaling parameter

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
        self.alpha = np.zeros((output_size, 1)) + 0.5
        self.train_alpha = train_alpha

        # Initialize network outputs for upper and lower nets
        self.upper_net = np.zeros((batch_size, output_size, 1))
        self.lower_net = np.zeros((batch_size, output_size, 1))

        # Used to store min-max reverse operation results
        self.minmax_reverse_stat = np.zeros((batch_size, output_size))

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
        params = np.size(self.upper_weight) * 2  # We have both upper and lower weights

        if self.use_bias:
            params += np.size(self.upper_bias) * 2  # Add bias parameters if enabled
        if self.train_alpha:
            params += np.size(self.alpha)  # Add alpha parameters if trainable

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
        
        input = input.reshape((-1, self.input_size))  # Ensure input is properly reshaped
        self.input = input

        # Ensure batch size matches the input batch size
        if self.batch_size < input.shape[0]:
            raise ValueError('Data batch size cannot be larger than model batch size')

        # Loop through each sample in the batch
        for batch_index, input_vector in enumerate(input):
            # Compute net input for upper and lower networks
            self.upper_net[batch_index] = self.upper_weight @ input_vector.reshape((-1, 1))
            self.lower_net[batch_index] = self.lower_weight @ input_vector.reshape((-1, 1))

            # Add bias if applicable
            if self.use_bias:
                self.upper_net[batch_index] += self.upper_bias
                self.lower_net[batch_index] += self.lower_bias

            # Apply activation function to the net inputs
            up_out = net2out(self.upper_net[batch_index], self.activation, alpha=self.alpha_activation)
            low_out = net2out(self.lower_net[batch_index], self.activation, alpha=self.alpha_activation)

            # Concatenate upper and lower outputs to find min and max
            concat_out = np.concatenate((up_out, low_out), axis=1)
            self.minmax_reverse_stat[batch_index] = np.argmax(concat_out)

            # Get max for upper and min for lower
            self.upper_output[batch_index] = np.max(concat_out, axis=1).reshape((-1, 1))
            self.lower_output[batch_index] = np.min(concat_out, axis=1).reshape((-1, 1))

            # Final output is a blend of upper and lower outputs, weighted by alpha
            self.final_output[batch_index] = self.alpha * self.upper_output[batch_index] + (1 - self.alpha) * self.lower_output[batch_index]

        # Return final output, reshaped to match (batch_size, output_size)
        return self.final_output[:input.shape[0]].reshape((-1, self.output_size))

    #################################################################

    def Adam_init(self):
        """
        Initializes the moment estimates for the Adam optimizer (mt and vt).
        """
        self.upper_weight_mt = np.zeros(self.upper_weight.shape)  # Moment estimates for upper weights
        self.lower_weight_mt = np.zeros(self.lower_weight.shape)  # Moment estimates for lower weights
        self.upper_weight_vt = np.zeros(self.upper_weight.shape)  # Velocity estimates for upper weights
        self.lower_weight_vt = np.zeros(self.lower_weight.shape)  # Velocity estimates for lower weights
        self.t = 0  # Time step counter for Adam updates

        # Initialize bias moments if applicable
        if self.use_bias:
            self.upper_bias_mt = np.zeros(self.upper_bias.shape)
            self.lower_bias_mt = np.zeros(self.lower_bias.shape)
            self.upper_bias_vt = np.zeros(self.upper_bias.shape)
            self.lower_bias_vt = np.zeros(self.lower_bias.shape)

        # Initialize alpha moments if trainable
        if self.train_alpha:
            self.alpha_mt = np.zeros(self.alpha.shape)
            self.alpha_vt = np.zeros(self.alpha.shape)

    #################################################################

    def update(self, grad_w_up: np.ndarray, grad_w_low: np.ndarray, grad_bias_up: np.ndarray, grad_bias_low: np.ndarray,
                grad_alpha: np.ndarray, method: str = 'Adam', learning_rate: float = 1e-3,
                   bias_learning_rate: float = 2e-4, adam_beta1: float = 0.9, adam_beta2: float = 0.99):
        """
        Updates the weights and biases using gradients from backpropagation. Supports Adam optimizer.
        
        Parameters:
        -----------
        grad_w_up : np.ndarray
            Gradient for the upper network weights.
        grad_w_low : np.ndarray
            Gradient for the lower network weights.
        grad_bias_up : np.ndarray
            Gradient for the upper network biases.
        grad_bias_low : np.ndarray
            Gradient for the lower network biases.
        grad_alpha : np.ndarray
            Gradient for the alpha parameter.
        method : str, optional
            Optimization method to use, default is 'Adam'.
        learning_rate : float, optional
            Learning rate for the weights, default is 1e-3.
        bias_learning_rate : float, optional
            Learning rate for the biases, default is 2e-4.
        adam_beta1 : float, optional
            Beta1 parameter for Adam optimizer, default is 0.9.
        adam_beta2 : float, optional
            Beta2 parameter for Adam optimizer, default is 0.99.
        """
        if method == 'Adam':
            # Adam optimizer parameters
            eps = 1e-7
            self.t += 1  # Increment timestep

            # Update moments for upper and lower weights
            self.upper_weight_mt = adam_beta1 * self.upper_weight_mt + (1 - adam_beta1) * grad_w_up
            self.upper_weight_vt = adam_beta2 * self.upper_weight_vt + (1 - adam_beta2) * np.square(grad_w_up)

            self.lower_weight_mt = adam_beta1 * self.lower_weight_mt + (1 - adam_beta1) * grad_w_low
            self.lower_weight_vt = adam_beta2 * self.lower_weight_vt + (1 - adam_beta2) * np.square(grad_w_low)

            # Bias-corrected moments
            m_hat_w_up = self.upper_weight_mt / (1 - adam_beta1 ** self.t)
            v_hat_w_up = self.upper_weight_vt / (1 - adam_beta2 ** self.t)

            m_hat_w_low = self.lower_weight_mt / (1 - adam_beta1 ** self.t)
            v_hat_w_low = self.lower_weight_vt / (1 - adam_beta2 ** self.t)

            # Compute weight updates
            delta_w_up = learning_rate * m_hat_w_up / (np.sqrt(v_hat_w_up) + eps)
            delta_w_low = learning_rate * m_hat_w_low / (np.sqrt(v_hat_w_low) + eps)

            # Update bias if enabled
            if self.use_bias:
                self.upper_bias_mt = adam_beta1 * self.upper_bias_mt + (1 - adam_beta1) * grad_bias_up
                self.upper_bias_vt = adam_beta2 * self.upper_bias_vt + (1 - adam_beta2) * np.square(grad_bias_up)

                self.lower_bias_mt = adam_beta1 * self.lower_bias_mt + (1 - adam_beta1) * grad_bias_low
                self.lower_bias_vt = adam_beta2 * self.lower_bias_vt + (1 - adam_beta2) * np.square(grad_bias_low)

                m_hat_bias_up = self.upper_bias_mt / (1 - adam_beta1 ** self.t)
                v_hat_bias_up = self.upper_bias_vt / (1 - adam_beta2 ** self.t)

                m_hat_bias_low = self.lower_bias_mt / (1 - adam_beta1 ** self.t)
                v_hat_bias_low = self.lower_bias_vt / (1 - adam_beta2 ** self.t)

                delta_bias_up = bias_learning_rate * m_hat_bias_up / (np.sqrt(v_hat_bias_up) + eps)
                delta_bias_low = bias_learning_rate * m_hat_bias_low / (np.sqrt(v_hat_bias_low) + eps)

            # Update alpha if trainable
            if self.train_alpha:
                self.alpha_mt = adam_beta1 * self.alpha_mt + (1 - adam_beta1) * grad_alpha
                self.alpha_vt = adam_beta2 * self.alpha_vt + (1 - adam_beta2) * np.square(grad_alpha)

                delta_alpha = learning_rate * self.alpha_mt / (np.sqrt(self.alpha_vt) + eps)

        else:
            # For non-Adam updates, use simple gradient descent
            delta_w_up = learning_rate * grad_w_up
            delta_w_low = learning_rate * grad_w_low
            if self.use_bias:
                delta_bias_up = bias_learning_rate * grad_bias_up
                delta_bias_low = bias_learning_rate * grad_bias_low
            if self.train_alpha:
                delta_alpha = learning_rate * grad_alpha

        # Apply the updates to weights and biases
        self.upper_weight -= delta_w_up
        self.lower_weight -= delta_w_low

        if self.use_bias:
            self.upper_bias -= delta_bias_up
            self.lower_bias -= delta_bias_low

        if self.train_alpha:
            self.alpha -= delta_alpha

    #################################################################

    def backward(self, error_batch: np.ndarray, method: str = 'Adam', 
                 learning_rate: float = 1e-3, bias_learning_rate: float = 2e-4, 
                 adam_beta1: float = 0.9, adam_beta2: float = 0.99) -> np.ndarray:
        """
        Performs backpropagation on the model, computing gradients for weights, biases, and alpha.
        
        Parameters:
        -----------
        error_batch : np.ndarray
            The batch of errors from the loss function.
        method : str, optional
            The optimization method to use (default is 'Adam').
        learning_rate : float, optional
            Learning rate for the weights (default is 1e-3).
        bias_learning_rate : float, optional
            Learning rate for the biases (default is 2e-4).
        adam_beta1 : float, optional
            Beta1 parameter for Adam optimizer (default is 0.9).
        adam_beta2 : float, optional
            Beta2 parameter for Adam optimizer (default is 0.99).
        
        Returns:
        --------
        np.ndarray
            The backpropagated error for the input layer.
        """

        error_out = np.zeros(self.input.shape)  # Initialize error output

        grad_w_up = np.zeros(self.upper_weight.shape)  # Gradient for upper weights
        grad_w_low = np.zeros(self.lower_weight.shape)  # Gradient for lower weights
        
        grad_bias_up = None
        grad_bias_low = None
        grad_alpha = None

        # Initialize bias gradients if necessary
        if self.use_bias:
            grad_bias_up = np.zeros(self.upper_bias.shape)
            grad_bias_low = np.zeros(self.lower_bias.shape)

        # Initialize alpha gradient if trainable
        if self.train_alpha:
            grad_alpha = np.zeros(self.alpha.shape)

        # Loop through each batch and compute sensitivities
        for batch_index, one_batch_error in enumerate(error_batch):
            # Compute gradient for alpha
            if self.train_alpha:
                grad_alpha += (self.upper_output[batch_index] - self.lower_output[batch_index])

            # Compute upper and lower network errors
            e_max = self.alpha * one_batch_error.reshape((-1, 1))
            e_min = (1 - self.alpha) * one_batch_error.reshape((-1, 1))

            e_upper = e_max * np.logical_not(self.minmax_reverse_stat[batch_index].reshape((-1, 1))) + \
                      e_min * self.minmax_reverse_stat[batch_index].reshape((-1, 1))
            e_lower = e_min * np.logical_not(self.minmax_reverse_stat[batch_index].reshape((-1, 1))) + \
                      e_max * self.minmax_reverse_stat[batch_index].reshape((-1, 1))

            # Compute sensitivity of upper and lower networks
            Fprime_up = net2Fprime(self.upper_net[batch_index], self.activation, self.alpha_activation)
            Fprime_low = net2Fprime(self.lower_net[batch_index], self.activation, self.alpha_activation)

            # Apply sensitivity (deltas)
            sensitivity_up = e_upper.reshape((-1, 1)) * Fprime_up
            sensitivity_low = e_lower.reshape((-1, 1)) * Fprime_low

            # Compute gradients for weights and biases
            grad_w_up += np.outer(sensitivity_up.ravel(), self.input[batch_index].ravel())
            grad_w_low += np.outer(sensitivity_low.ravel(), self.input[batch_index].ravel())

            if self.use_bias:
                grad_bias_up += sensitivity_up
                grad_bias_low += sensitivity_low

            # Propagate error back to previous layer
            error_out[batch_index] = np.ravel(self.upper_weight.T @ sensitivity_up + self.lower_weight.T @ sensitivity_low)

        # Normalize gradients
        grad_w_up /= error_batch.shape[0]
        grad_w_low /= error_batch.shape[0]

        if self.use_bias:
            grad_bias_up /= error_batch.shape[0]
            grad_bias_low /= error_batch.shape[0]

        if self.train_alpha:
            grad_alpha /= error_batch.shape[0]

        # Update model parameters based on gradients
        self.update(grad_w_up, grad_w_low, grad_bias_up, grad_bias_low, grad_alpha,
                    method=method, learning_rate=learning_rate,
                    bias_learning_rate=bias_learning_rate, adam_beta1=adam_beta1, adam_beta2=adam_beta2)

        return error_out