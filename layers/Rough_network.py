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
    """

    def __init__(self, input_size: int, output_size: int, use_bias: bool = True, batch_size: int = 32,
                 train_weights: bool = True, train_bias: bool = True, train_blending: bool = False,
                 activation: str = 'sigmoid', alpha_acti: float = None, weights_uniform_range: tuple = (-1, 1)):

        self.output_size = output_size  # Number of output neurons
        self.input_size = input_size  # Number of input neurons/features
        self.batch_size = batch_size  # Batch size
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
            self.final_output[batch_index] = self.blending_factor * self.upper_output[batch_index] + (1 - self.blending_factor) * self.lower_output[batch_index]

        # Return final output, reshaped to match (batch_size, output_size)
        return self.final_output[:input.shape[0]].reshape((-1, self.output_size))

    #################################################################

    def optimzer_init(self, optimizer='Adam', **kwargs) -> None:
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, grads: np.ndarray, learning_rate: float = 1e-3) -> None:
        deltas = self.Optimizer(grads, learning_rate)
        ind2 = 0
        if self.train_weights:
            ind1 = ind2
            ind2 += int(np.size(self.upper_weight))
            delta_w = deltas[ind1:ind2].reshape(self.upper_weight.shape)
            self.upper_weight -= delta_w
            ind1 = ind2
            ind2 += int(np.size(self.lower_weight))
            delta_w = deltas[ind1:ind2].reshape(self.lower_weight.shape)
            self.lower_weight -= delta_w
        if self.train_bias:
            ind1 = ind2
            ind2 += np.size(self.upper_bias)
            delta_bias = deltas[ind1:ind2].reshape(self.upper_bias.shape)
            self.upper_bias -= delta_bias
            ind1 = ind2
            ind2 += np.size(self.lower_bias)
            delta_bias = deltas[ind1:ind2].reshape(self.lower_bias.shape)
            self.lower_bias -= delta_bias
        if self.train_blending:
            ind1 = ind2
            ind2 += np.size(self.blending_factor)
            delta_blend = deltas[ind1:ind2].reshape(self.blending_factor.shape)
            self.blending_factor -= delta_blend

    #################################################################

    def backward(self, error_batch: np.ndarray, learning_rate: float = 1e-3, return_error=False, return_grads=False, modify=True):
        if return_error:
            error_in = np.zeros(self.input.shape)  # Initialize error output

        # Initialize the gradients for weights and biases
        grad_w_up = np.zeros(self.upper_weight.shape) if self.train_weights else None
        grad_w_low = np.zeros(self.lower_weight.shape) if self.train_weights else None
        grad_bias_up = np.zeros(self.upper_bias.shape) if self.train_bias else None
        grad_bias_low = np.zeros(self.lower_bias.shape) if self.train_bias else None
        grad_alpha = np.zeros(self.blending_factor.shape) if self.train_blending else None

        # Loop through each batch and compute sensitivities
        for batch_index, one_batch_error in enumerate(error_batch):
            # Compute gradient for alpha
            if self.train_blending:
                grad_alpha += one_batch_error.reshape((-1, 1)) * \
                    (self.upper_output[batch_index] - self.lower_output[batch_index])

            # Compute upper and lower network errors
            e_max = self.blending_factor * one_batch_error.reshape((-1, 1))
            e_min = (1 - self.blending_factor) * one_batch_error.reshape((-1, 1))

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
            if self.train_weights:
                grad_w_up += np.outer(sensitivity_up.ravel(), self.input[batch_index].ravel())
                grad_w_low += np.outer(sensitivity_low.ravel(), self.input[batch_index].ravel())

            if self.train_bias:
                grad_bias_up += sensitivity_up
                grad_bias_low += sensitivity_low

            # Propagate error back to previous layer
            if return_error:
                error_in[batch_index] = np.ravel(self.upper_weight.T @ sensitivity_up + self.lower_weight.T @ sensitivity_low)

        # Normalize gradients
        if self.train_weights:
            grad_w_up /= error_batch.shape[0]
            grad_w_low /= error_batch.shape[0]

        if self.use_bias:
            grad_bias_up /= error_batch.shape[0]
            grad_bias_low /= error_batch.shape[0]

        if self.train_blending:
            grad_alpha /= error_batch.shape[0]

        # Update the parameters using the computed gradients
        grads = None if (grad_w_up is None) and (grad_bias_up is None) else np.array([]).reshape((-1,1))
        if grads is not None:
            if grad_w_up is not None:
                grads = np.concatenate((grads, grad_w_up.reshape((-1,1))))
                grads = np.concatenate((grads, grad_w_low.reshape((-1,1))))
            if grad_bias_up is not None:
                grads = np.concatenate((grads, grad_bias_up.reshape((-1,1))))
                grads = np.concatenate((grads, grad_bias_low.reshape((-1,1))))
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
