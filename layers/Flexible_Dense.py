import numpy as np
from activations.flexible_activation_functions import net2out, net2Fprime, net2Fstar
from initializers.weight_initializer import Dense_weight_init
from optimizers.set_optimizer import init_optimizer


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
    - train_weights (bool): Whether to train weights or not
    - train_bias (bool): Whether to train bias or not
    - train_alpha (bool): Whether to train alpha or not
    - train_lambda (bool): Whether to train lambda or not
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
                 activation: str = 'leaky_relu', alpha: float = None, lambda_=None,
                 train_weights: bool = True, train_bias: bool = True, train_alpha: bool = True, train_lambda: bool = True,
                 weights_init_method: str = 'he', 
                 weight_distribution: str = 'normal', orthogonal_scale_factor: float = 1.0, 
                 weights_uniform_range: tuple = None):
        self.output_size = output_size
        self.input_size = input_size
        self.batch_size = batch_size
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
        #if self.activation in ['selu', 'elu']:
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
            # Calculate derivatives for alpha and lambda if required
            if self.train_alpha or self.train_lambda:
                Fstar = net2Fstar(self.net[batch_index], self.activation, self.alpha, lambda_param=self.lambda_param)
                # Update alpha and lambda gradients based on activation function
                if self.activation in ['selu', 'elu']:
                    if self.train_alpha:
                        grad_alpha += one_batch_error.reshape((-1, 1)) * Fstar[0]
                    if self.train_lambda:
                        grad_lambda += one_batch_error.reshape((-1, 1)) * Fstar[1]
                else:
                    if self.train_alpha:
                        grad_alpha += one_batch_error.reshape((-1, 1)) * Fstar

            # Compute derivative of the activation function
            Fprime = net2Fprime(self.net[batch_index], self.activation, self.alpha, lambda_param=self.lambda_param)

            # Calculate sensitivity as the product of the error and activation derivative
            sensitivity = one_batch_error.reshape((-1, 1)) * Fprime

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
        if self.train_alpha:
            grad_alpha /= error_batch.shape[0]
        if self.train_bias:
            grad_bias /= error_batch.shape[0]
        if self.train_lambda:
            grad_lambda /= error_batch.shape[0]

        # Collect gradients into a single array if not None
        grads = None if (grad_w is None) and (grad_bias is None) and (grad_alpha is None) and (grad_lambda is None) else \
            np.array([]).reshape((-1,1))
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
