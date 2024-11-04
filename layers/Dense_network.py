import numpy as np
from activations.activation_functions import net2out, net2Fprime
from initializers.weight_initializer import Dense_weight_init
from optimizers.set_optimizer import init_optimizer


class Dense:
    def __init__(self, input_size: int, output_size: int, use_bias: bool = True, train_bias: bool = True,
                 train_weights: bool = True, batch_size: int = 32, 
                 activation: str = 'relu', alpha: float = None, weights_init_method: str = 'he', 
                 weight_distribution: str = 'normal', orthogonal_scale_factor: float = 1.0, 
                 weights_uniform_range: tuple = None):
        """
        Initializes a dense (fully connected) layer.
        
        Parameters:
        input_size (int): Number of input neurons.
        output_size (int): Number of output neurons.
        use_bias (bool): Whether to use bias in the layer. Default is True.
        train_bias (bool): Whether to train bias if use_bias or not
        train_weights (bool): Whether to train weights or not
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
        self.train_bias = False if use_bias is False else train_bias
        self.train_weights = train_weights
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
        params = 0
        if self.train_bias:
            params += np.size(self.bias)
        if self.train_weights:
            params += np.size(self.weight)
        return params
    
    #################################################################

    def all_params(self) -> int:
        """
        Returns the number of trainable and non trainable parameters in the layer.
        
        Returns:
        int: Total number of all parameters (weights and biases).
        """
        params = np.size(self.weight)
        if self.use_bias:
            params += np.size(self.bias)
        return params

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
    
    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for the layer using the specified method.

        Parameters:
        -----------
        optimizer : str, optional
            The optimization algorithm to use (default is 'Adam').
        **kwargs : dict, optional
            Additional parameters for the optimizer configuration.
            
        Returns:
        --------
        None
        """
        # Initializes optimizer instance with the specified method and configuration,
        # setting it to the layer's trainable parameters
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, grads: np.ndarray, learning_rate: float = 1e-3) -> None:
        """
        Updates the layer's weights and biases based on calculated gradients.

        Parameters:
        -----------
        grads : np.ndarray
            Gradients of the loss with respect to each trainable parameter in the layer.
        learning_rate : float, optional
            Step size used for each iteration of parameter updates (default is 1e-3).
            
        Returns:
        --------
        None
        """
        # Get parameter update values (deltas) from optimizer instance
        deltas = self.Optimizer(grads, learning_rate)
        
        # Initialize the index for parameter updates
        ind2 = 0

        # Update weights if the layer's weights are trainable
        if self.train_weights:
            ind1 = ind2
            ind2 += int(np.size(self.weight))
            delta_w = deltas[ind1:ind2].reshape(self.weight.shape)
            self.weight -= delta_w  # Apply weight update

        # Update bias if the layer's bias is trainable
        if self.train_bias:
            ind1 = ind2
            ind2 += np.size(self.bias)
            delta_bias = deltas[ind1:ind2].reshape(self.bias.shape)
            self.bias -= delta_bias  # Apply bias update

    #################################################################

    def backward(self, error_batch: np.ndarray, learning_rate: float = 1e-3, 
                return_error: bool = False, return_grads: bool = False, modify: bool = True):
        """
        Computes gradients for weights and biases and optionally updates parameters and returns errors and gradients.

        Parameters:
        -----------
        error_batch : np.ndarray
            Batch of errors from the subsequent layer, with shape (batch_size, output_size).
        learning_rate : float, optional
            Step size for parameter updates (default is 1e-3).
        return_error : bool, optional
            If True, returns the error gradients with respect to the inputs (default is False).
        return_grads : bool, optional
            If True, returns gradients of the weights and biases (default is False).
        modify : bool, optional
            If True, updates weights and biases (default is True).
            
        Returns:
        --------
        dict or np.ndarray or None
            Returns a dictionary with `error_in` and `gradients` if both `return_error` and `return_grads` are True.
            Returns `error_in` if `return_error` is True and `return_grads` is False.
            Returns `gradients` if `return_grads` is True and `return_error` is False.
        """
        # If error gradient w.r.t inputs is required, initialize error_in array
        if return_error:
            error_in = np.zeros(self.input.shape)

        # Initialize gradients for weights and biases
        grad_w = np.zeros(self.weight.shape) if self.train_weights else None
        grad_bias = np.zeros(self.bias.shape) if self.train_bias else None

        # Process each error in the batch
        for batch_index, one_batch_error in enumerate(error_batch):
            # Compute the derivative of the activation function at each net input
            Fprime = net2Fprime(self.net[batch_index], self.activation, self.alpha_activation)
            
            # Calculate sensitivity based on activation function type
            if self.activation == 'softmax':
                sensitivity = (Fprime @ one_batch_error).reshape((-1, 1))
            else:
                sensitivity = one_batch_error.reshape((-1, 1)) * Fprime

            # Accumulate weight gradient if trainable
            if self.train_weights:
                grad_w += np.outer(sensitivity.ravel(), self.input[batch_index].ravel())

            # Accumulate bias gradient if trainable
            if self.train_bias:
                grad_bias += sensitivity

            # Compute the input error gradient if required
            if return_error:
                error_in[batch_index] = np.ravel(self.weight.T @ sensitivity)

        # Average gradients over the batch if trainable
        if self.train_weights:
            grad_w /= error_batch.shape[0]
        if self.train_bias:
            grad_bias /= error_batch.shape[0]
        
        # Combine weight and bias gradients into one array if needed for update
        grads = None if (grad_w is None) and (grad_bias is None) else np.array([]).reshape((-1, 1))
        if grads is not None:
            if grad_w is not None:
                grads = np.concatenate((grads, grad_w.reshape((-1, 1))))
            if grad_bias is not None:
                grads = np.concatenate((grads, grad_bias.reshape((-1, 1))))

        # Update parameters if modify is True
        if modify:
            self.update(grads, learning_rate=learning_rate)

        # Return gradients or error gradients based on flags
        if return_error and return_grads:
            return {'error_in': error_in, 'gradients': grads}
        elif return_error:
            return error_in
        elif return_grads:
            return grads
