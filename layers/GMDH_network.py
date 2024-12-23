import numpy as np
from itertools import combinations
from activations.activation_functions import net2out, net2Fprime
from initializers.weight_initializer import Dense_weight_init
from optimizers.set_optimizer import init_optimizer


class GMDH:
    """
    Group Method of Data Handling (GMDH) implementation with parallel Adaline blocks.

    Parameters:
    -----------
    input_size : int
        Number of input features.
    use_bias : bool, optional
        Whether to include a bias term in the model (default is True).
    train_bias : bool, optional
        Whether to train the bias term (default is True).
    train_weights : bool, optional
        Whether to train the weights (default is True).
    batch_size : int, optional
        Number of samples per batch (default is 32).
    weights_init_method : str, optional
        Method to initialize the weights (default is 'uniform').
    weight_distribution : str, optional
        Distribution type for weights (default is 'uniform').
    orthogonal_scale_factor : float, optional
        Scaling factor for orthogonal weight initialization (default is 1.0).
    weights_uniform_range : tuple, optional
        Range for uniform weight initialization (default is None).
    L2_coe : float, optional
        L2 regularization coefficient (default is 0.0).
    L1_coe : float, optional
        L1 regularization coefficient (default is 0.0).

    Attributes:
    -----------
    weight : np.ndarray
        Weight matrix of the model.
    bias : np.ndarray
        Bias vector of the model.
    output : np.ndarray
        Model output for the current batch.
    input_inds : np.ndarray
        Array storing all possible combinations of input indices.
    """

    def __init__(self, input_size: int, use_bias: bool = True, train_bias: bool = True,
                 train_weights: bool = True, batch_size: int = 32,
                 weights_init_method: str = 'uniform',
                 weight_distribution: str = 'uniform', orthogonal_scale_factor: float = 1.0,
                 weights_uniform_range: tuple = None, L2_coe: float = 0.0, L1_coe: float = 0.0) -> None:
        # Ensure that input_size is at least 2
        if input_size < 2:
            raise ValueError('GMDH input size cannot be less than 2')

        # Initialize model parameters
        self.input_size = input_size
        self.output_size = np.math.factorial(input_size) // (np.math.factorial(2) * np.math.factorial(input_size - 2))
        self.batch_size = batch_size
        self.use_bias = use_bias
        self.train_bias = False if use_bias is False else train_bias
        self.train_weights = train_weights
        self.activation = 'None'
        self.L2_coe = L2_coe
        self.L1_coe = L1_coe

        # Initialize weights
        self.weight = Dense_weight_init(5, self.output_size, method=weights_init_method,
                                        distribution=weight_distribution, scale_factor=orthogonal_scale_factor,
                                        ranges=weights_uniform_range)

        # Initialize bias if applicable
        if self.use_bias:
            self.bias = np.zeros((self.output_size, 1))

        # Initialize model outputs and input indices
        self.output = np.zeros((batch_size, self.output_size, 1))
        self.input_inds = np.array(list(combinations(range(input_size), 2)))

    #################################################################

    def trainable_params(self) -> int:
        """
        Returns the number of trainable parameters in the layer.

        Returns:
        --------
        int
            Total number of trainable parameters (weights and biases).
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
        Returns the total number of parameters in the layer (trainable and non-trainable).

        Returns:
        --------
        int
            Total number of parameters (weights and biases).
        """
        params = np.size(self.weight)
        if self.use_bias:
            params += np.size(self.bias)
        return params

    #################################################################

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation for the model.

        Parameters:
        -----------
        input : np.ndarray
            Input data of shape (batch_size, input_size).

        Returns:
        --------
        np.ndarray
            Model output of shape (batch_size, output_size).
        """
        # Validate input dimensions
        self.input = input
        if self.batch_size < input.shape[0]:
            raise ValueError('Data batch size cannot be larger than model batch size')

        # Iterate over each batch sample
        for batch_index, input_vector in enumerate(input):
            # Extract input combinations for Adaline blocks
            adalines_inputs = input_vector[self.input_inds]

            # Compute output using GMDH polynomial terms
            self.output[batch_index] = (
                adalines_inputs[:, 0] * self.weight[:, 0] +
                np.square(adalines_inputs[:, 0]) * self.weight[:, 1] +
                adalines_inputs[:, 0] * adalines_inputs[:, 1] * self.weight[:, 2] +
                np.square(adalines_inputs[:, 1]) * self.weight[:, 3] +
                adalines_inputs[:, 1] * self.weight[:, 4]
            ).reshape((-1, 1))

            # Add bias if applicable
            if self.use_bias:
                self.output[batch_index] += self.bias

        # Return the computed outputs
        batch_index += 1
        return self.output[:batch_index, :, 0]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initialize the optimizer for updating model parameters.

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
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, grads: np.ndarray, learning_rate: float = 1e-3) -> None:
        """
        Update model parameters using calculated gradients.

        Parameters:
        -----------
        grads : np.ndarray
            Gradients with respect to the trainable parameters.
        learning_rate : float, optional
            Step size for parameter updates (default is 1e-3).

        Returns:
        --------
        None
        """
        # Compute parameter updates using the optimizer
        deltas = self.Optimizer(grads, learning_rate)
        ind2 = 0

        # Update weights
        if self.train_weights:
            ind1 = ind2
            ind2 += int(np.size(self.weight))
            delta_w = deltas[ind1:ind2].reshape(self.weight.shape)
            self.weight -= delta_w

        # Update bias
        if self.train_bias:
            ind1 = ind2
            ind2 += np.size(self.bias)
            delta_bias = deltas[ind1:ind2].reshape(self.bias.shape)
            self.bias -= delta_bias

    #################################################################

    def backward(self, error_batch: np.ndarray, learning_rate: float = 1e-3,
                 return_error: bool = False, return_grads: bool = False, modify: bool = True):
        """
        Perform backpropagation to calculate gradients and optionally update weights.

        Parameters:
        -----------
        error_batch : np.ndarray
            Error signals from the next layer, shape (batch_size, output_size).
        learning_rate : float, optional
            Learning rate for updating parameters (default is 1e-3).
        return_error : bool, optional
            Whether to return propagated error (default is False).
        return_grads : bool, optional
            Whether to return calculated gradients (default is False).
        modify : bool, optional
            Whether to update weights and biases during this call (default is True).

        Returns:
        --------
        dict or np.ndarray or None
            A dictionary containing errors and gradients, or only errors, or only gradients, depending on flags.
        """
        if return_error:
            error_in = np.zeros(self.input.shape)

        # Initialize gradients
        grad_w = np.zeros(self.weight.shape) if self.train_weights else None
        grad_bias = np.zeros(self.bias.shape) if self.train_bias else None

        # Iterate over each batch sample
        for batch_index, one_batch_error in enumerate(error_batch):
            input_vector = self.input[batch_index]
            adalines_inputs = input_vector[self.input_inds]

            # Compute weight gradients
            if self.train_weights:
                grad_w1 = adalines_inputs[:, 0] * one_batch_error
                grad_w5 = adalines_inputs[:, 1] * one_batch_error
                grad_w2 = np.square(adalines_inputs[:, 0]) * one_batch_error
                grad_w4 = np.square(adalines_inputs[:, 1]) * one_batch_error
                grad_w3 = adalines_inputs[:, 0] * adalines_inputs[:, 1] * one_batch_error
                grad_w += np.concatenate((grad_w1.reshape((-1, 1)),
                                          grad_w2.reshape((-1, 1)),
                                          grad_w3.reshape((-1, 1)),
                                          grad_w4.reshape((-1, 1)),
                                          grad_w5.reshape((-1, 1))), axis=1)

            # Compute bias gradients
            if self.train_bias:
                grad_bias += one_batch_error.reshape((-1,1))

            # Compute propagated error for inputs
            if return_error:
                grad_x1 = (self.weight[:, 0] + 2 * self.weight[:, 1] * adalines_inputs[:, 0] + self.weight[:, 2] * adalines_inputs[:, 1]) * one_batch_error
                grad_x2 = (self.weight[:, 4] + 2 * self.weight[:, 3] * adalines_inputs[:, 1] + self.weight[:, 2] * adalines_inputs[:, 0]) * one_batch_error
                grad_x = np.concatenate((grad_x1.reshape((-1, 1)),
                                         grad_x2.reshape((-1, 1))), axis=1)
                error_in_batch = np.zeros(self.input_size)
                np.add.at(error_in_batch, self.input_inds.ravel(), grad_x.ravel())
                error_in[batch_index] = error_in_batch

        # Average and regularize weight gradients
        if self.train_weights:
            grad_w /= error_batch.shape[0]
            grad_w += self.L1_coe * np.sign(self.weight) + self.L2_coe * self.weight

        # Average bias gradients
        if self.train_bias:
            grad_bias /= error_batch.shape[0]

        # Compile gradients into a single array
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
        if grads is not None:
            if grad_w is not None:
                grads = np.concatenate((grads, grad_w.reshape((-1, 1))))
            if grad_bias is not None:
                grads = np.concatenate((grads, grad_bias.reshape((-1, 1))))

        # Update parameters if modify flag is set
        if modify:
            self.update(grads, learning_rate=learning_rate)

        # Return requested values
        if return_error and return_grads:
            return {'error_in': error_in, 'gradients': grads}
        elif return_error:
            return error_in
        elif return_grads:
            return grads
