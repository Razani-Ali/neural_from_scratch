import numpy as np
from activations.activation_functions import net2out, net2Fprime
from initializers.weight_initializer import Dense_weight_init
from optimizers.set_optimizer import init_optimizer


class Dense:
    def __init__(self, input_size: int, output_size: int, use_bias: bool = True, train_bias: bool = True,
                 train_weights: bool = True, batch_size: int = 32, 
                 activation: str = 'relu', alpha: float = None, weights_init_method: str = 'he', 
                 weight_distribution: str = 'normal', orthogonal_scale_factor: float = 1.0, 
                 weights_uniform_range: tuple = None, L2_coe: float = 0.0, L1_coe: float = 0.0):
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
        L2_coe (float, optional): L2 regularization coefficient
        L1_coe (float, optional): L1 regularization coefficient
        """
        self.output_size = output_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.use_bias = use_bias
        self.train_bias = False if use_bias is False else train_bias
        self.train_weights = train_weights
        self.activation = activation
        self.L2_coe = L2_coe
        self.L1_coe = L1_coe
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
            one_batch_error = one_batch_error.reshape((-1,1))

            # Compute the derivative of the activation function at each net input
            Fprime = net2Fprime(self.net[batch_index], self.activation, self.alpha_activation)
            
            # Calculate sensitivity based on activation function type
            if self.activation == 'softmax':
                sensitivity = Fprime @ one_batch_error
            else:
                sensitivity = one_batch_error * Fprime

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
            grad_w += self.L1_coe * np.sign(self.weight) + self.L2_coe * self.weight
        if self.train_bias:
            grad_bias /= error_batch.shape[0]
        
        # Combine weight and bias gradients into one array if needed for update
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
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

    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class TimeDense:
    def __init__(self, time_steps: int, input_size: int, output_size: int, use_bias: bool = True, train_bias: bool = True,
                 train_weights: bool = True, batch_size: int = 32, 
                 activation: str = 'relu', alpha: float = None, weights_init_method: str = 'he', 
                 weight_distribution: str = 'normal', orthogonal_scale_factor: float = 1.0, 
                 weights_uniform_range: tuple = None, L2_coe: float = 0.0, L1_coe: float = 0.0):
        """
        Initializes a dense (fully connected) layer.

        Parameters:
            time_steps (int): Number of time steps in input sequences.
            input_size (int): Number of input features per sequence.
            output_size (int): Number of neurons in the dense layer.
            use_bias (bool): Whether to include bias in the layer. Defaults to True.
            train_bias (bool): Whether to train bias parameters if use_bias is enabled. Defaults to True.
            train_weights (bool): Whether to train weights of the layer. Defaults to True.
            batch_size (int): Number of sequences per training batch. Defaults to 32.
            activation (str): Name of the activation function to use. Defaults to 'relu'.
            alpha (float): Optional parameter for activation functions requiring a slope or scaling (e.g., Leaky ReLU).
            weights_init_method (str): Method to initialize weights ('he', 'xavier', etc.). Defaults to 'he'.
            weight_distribution (str): Probability distribution for weights ('normal' or 'uniform'). Defaults to 'normal'.
            orthogonal_scale_factor (float): Scaling factor for orthogonal weight initialization. Defaults to 1.0.
            weights_uniform_range (tuple): Range for weights if using uniform distribution. Optional.
            L2_coe (float): L2 regularization coefficient for weights. Defaults to 0.0.
            L1_coe (float): L1 regularization coefficient for weights. Defaults to 0.0.
        """
        self.output_size = output_size  # Number of neurons in the dense layer
        self.input_size = input_size    # Number of input features
        self.batch_size = batch_size    # Batch size for input sequences
        self.time_steps = time_steps    # Number of time steps in sequences
        self.use_bias = use_bias        # Flag to determine if bias is used
        self.train_bias = train_bias if use_bias else False  # Trainable bias only if use_bias is True
        self.train_weights = train_weights  # Flag to determine if weights are trainable
        self.activation = activation    # Activation function name
        self.L2_coe = L2_coe            # L2 regularization coefficient
        self.L1_coe = L1_coe            # L1 regularization coefficient
        self.alpha_activation = alpha   # Parameter for advanced activation functions

        # Weight initialization
        self.weight = Dense_weight_init(input_size, output_size, 
                                        method=weights_init_method, 
                                        distribution=weight_distribution, 
                                        scale_factor=orthogonal_scale_factor, 
                                        ranges=weights_uniform_range)

        # Initialize biases if they are used
        if self.use_bias:
            self.bias = np.zeros((output_size, 1))  # Bias initialized to zero
        
        # Gradients for weights and biases
        self.grad_w = np.zeros(self.weight.shape) if self.train_weights else None  # Gradient of weights
        self.grad_b = np.zeros(self.bias.shape) if self.train_bias else None  # Gradient of biases
        
        # Input, net (pre-activation), and output tensors for forward pass
        self.input = np.zeros((batch_size, time_steps, input_size, 1))  # Input tensor
        self.net = np.zeros((batch_size, time_steps, output_size, 1))   # Pre-activation values
        self.output = np.zeros((batch_size, time_steps, output_size, 1))  # Post-activation values

    #################################################################

    def trainable_params(self) -> int:
        """
        Computes the total number of trainable parameters in the layer.

        Returns:
            int: The total number of trainable weights and biases.
        """
        params = 0
        if self.train_bias:  # Add bias parameters if they are trainable
            params += np.size(self.bias)
        if self.train_weights:  # Add weight parameters if they are trainable
            params += np.size(self.weight)
        return params
    
    #################################################################

    def all_params(self) -> int:
        """
        Computes the total number of all parameters (trainable and non-trainable).

        Returns:
            int: The total number of weights and biases in the layer.
        """
        params = np.size(self.weight)  # Count all weight parameters
        if self.use_bias:  # Add bias parameters if bias is used
            params += np.size(self.bias)
        return params

    #################################################################

    def __call__(self, batch_index: int, seq_index: int, input: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass for a single sequence within a batch.

        Parameters:
            batch_index (int): Index of the batch being processed.
            seq_index (int): Index of the sequence within the batch.
            input (np.ndarray): Input array for the sequence.

        Returns:
            np.ndarray: Output of the layer after applying the activation function.
        """
        self.input[batch_index, seq_index] = input  # Store input

        if self.batch_size < batch_index:  # Validate batch size
            raise ValueError("Batch index exceeds batch size.")
        
        # Compute net input to the layer
        self.net[batch_index, seq_index] = self.weight @ input
        if self.use_bias:  # Add bias if applicable
            self.net[batch_index, seq_index] += self.bias
        
        # Apply activation function
        self.output[batch_index, seq_index] = net2out(
            self.net[batch_index, seq_index], 
            self.activation, 
            alpha=self.alpha_activation
        )
        return self.output[batch_index, seq_index]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for the layer.

        Parameters:
            optimizer (str): Name of the optimization algorithm (e.g., 'Adam', 'SGD').
            **kwargs: Additional parameters to configure the optimizer.
        """
        # Initialize optimizer instance and assign it to the trainable parameters
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, batch_size: int, learning_rate: float = 1e-3, grads: np.ndarray = None) -> None:
        """
        Updates weights and biases using gradients computed during backpropagation.

        Parameters:
            batch_size (int): Number of sequences in the current batch.
            learning_rate (float): Learning rate for parameter updates. Defaults to 1e-3.
            grads (np.ndarray): Gradients with respect to parameters. If None, regularization is applied.

        Returns:
            None
        """
        if grads is None:
            # Apply L1 and L2 regularization to weight gradients if applicable
            if self.train_weights:
                self.grad_w += (self.L1_coe * np.sign(self.weight) + self.L2_coe * self.weight)

            # Combine gradients into a single array
            grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
            if grads is not None:
                if self.grad_w is not None:
                    grads = np.concatenate((grads, self.grad_w.reshape((-1, 1))))
                if self.grad_b is not None:
                    grads = np.concatenate((grads, self.grad_b.reshape((-1, 1))))
                grads /= batch_size  # Average gradients over the batch

        # Use optimizer to compute parameter updates (deltas)
        deltas = self.Optimizer(grads, learning_rate)

        # Apply updates to weights and biases
        ind2 = 0
        if self.train_weights:
            ind1 = ind2
            ind2 += int(np.size(self.weight))
            delta_w = deltas[ind1:ind2].reshape(self.weight.shape)
            self.weight -= delta_w  # Update weights
        if self.train_bias:
            ind1 = ind2
            ind2 += np.size(self.bias)
            delta_bias = deltas[ind1:ind2].reshape(self.bias.shape)
            self.bias -= delta_bias  # Update biases

        # Reset gradients to zero for the next training iteration
        self.grad_w = self.grad_w * 0 if self.train_weights else None
        self.grad_b = self.grad_b * 0 if self.train_bias else None

    #################################################################

    def return_grads(self) -> np.ndarray:
        """
        Retrieves the gradients for weights and biases.

        Returns:
            np.ndarray: Combined gradients for all trainable parameters.
        """
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
        if self.grad_w is not None:
            grad_w = self.grad_w + (self.L1_coe * np.sign(self.weight) + self.L2_coe * self.weight)
            grads = np.concatenate((grads, grad_w.reshape((-1, 1))))
        if self.grad_b is not None:
            grads = np.concatenate((grads, self.grad_b.reshape((-1, 1))))
        return grads

    #################################################################

    def backward(self, batch_index: int, seq_index: int, error: np.ndarray) -> np.ndarray:
        """
        Performs backpropagation for a single sequence, computing gradients and propagating the error backward.

        Parameters:
            batch_index (int): Index of the batch being processed.
            seq_index (int): Index of the sequence within the batch.
            error (np.ndarray): Error propagated from the subsequent layer.

        Returns:
            np.ndarray: Error propagated to the previous layer.
        """
        # Compute activation derivative (Fprime) for the current sequence and batch
        Fprime = net2Fprime(self.net[batch_index, seq_index], self.activation, self.alpha_activation)

        # Compute sensitivity of the current layer
        if self.activation == 'softmax':
            sensitivity = Fprime @ error
        else:
            sensitivity = error * Fprime

        # Accumulate gradients for weights if trainable
        if self.train_weights:
            self.grad_w += np.outer(sensitivity.ravel(), self.input[batch_index, seq_index].ravel())

        # Accumulate gradients for biases if trainable
        if self.train_bias:
            self.grad_b += sensitivity

        # Compute error to propagate to the previous layer
        error_in = self.weight.T @ sensitivity

        return error_in.reshape((-1, 1))
    
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class Dense1Feedback:
    def __init__(self, time_steps: int, input_size: int, output_size: int, feedback_size: int = None,
                 use_bias: bool = True, train_bias: bool = True, train_weights: bool = True, batch_size: int = 32, 
                 activation: str = 'relu', alpha: float = None, weights_init_method: str = 'he', 
                 weight_distribution: str = 'normal', orthogonal_scale_factor: float = 1.0, 
                 weights_uniform_range: tuple = None, L2_coe: float = 0.0, L1_coe: float = 0.0):
        """
        Initializes a dense (fully connected) layer with feedback.

        Parameters:
            time_steps (int): Number of time steps or sequences in the input data.
            input_size (int): Number of input features (size of the input vector).
            output_size (int): Number of output features (size of the output vector).
            feedback_size (int, optional): Size of the feedback (state) vector. Defaults to output_size if not provided.
            use_bias (bool): Whether to include bias terms in the layer. Defaults to True.
            train_bias (bool): Whether to allow training of bias parameters. Defaults to True.
            train_weights (bool): Whether to allow training of weight parameters. Defaults to True.
            batch_size (int): Number of sequences per training batch. Defaults to 32.
            activation (str): Name of the activation function to use (e.g., 'relu', 'sigmoid'). Defaults to 'relu'.
            alpha (float, optional): Parameter for activation functions like Leaky ReLU and ELU. Defaults to None.
            weights_init_method (str): Method for initializing weights ('he', 'xavier', etc.). Defaults to 'he'.
            weight_distribution (str): Distribution type for weights ('normal', 'uniform'). Defaults to 'normal'.
            orthogonal_scale_factor (float): Scaling factor for orthogonal initialization. Defaults to 1.0.
            weights_uniform_range (tuple, optional): Range for uniform weight initialization. Defaults to None.
            L2_coe (float, optional): Coefficient for L2 regularization. Defaults to 0.0.
            L1_coe (float, optional): Coefficient for L1 regularization. Defaults to 0.0.
        """
        self.output_size = output_size  # Size of the output vector
        self.input_size = input_size    # Size of the input vector
        self.feedback_size = feedback_size if feedback_size is not None else output_size  # Feedback vector size
        self.batch_size = batch_size    # Number of sequences in a batch
        self.time_steps = time_steps    # Number of time steps in the sequence
        self.use_bias = use_bias        # Flag to include bias terms
        self.train_bias = False if use_bias is False else train_bias  # Enable training bias if bias is used
        self.train_weights = train_weights  # Flag to enable training of weights
        self.activation = activation    # Activation function name
        self.L2_coe = L2_coe            # Coefficient for L2 regularization
        self.L1_coe = L1_coe            # Coefficient for L1 regularization
        self.alpha_activation = alpha   # Optional alpha value for activation functions

        # Initialize weights for input and feedback
        self.weight = Dense_weight_init(input_size, output_size, method=weights_init_method, 
                                        distribution=weight_distribution, scale_factor=orthogonal_scale_factor, 
                                        ranges=weights_uniform_range)
        self.weight_state = Dense_weight_init(self.feedback_size, output_size, method=weights_init_method, 
                                              distribution=weight_distribution, scale_factor=orthogonal_scale_factor, 
                                              ranges=weights_uniform_range)
        
        # Initialize bias if applicable
        if self.use_bias:
            self.bias = np.zeros((output_size, 1))  # Initialize biases to zero
        
        # Initialize gradients for weights and biases
        self.grad_w = np.zeros(self.weight.shape) if self.train_weights else None
        self.grad_w_state = np.zeros(self.weight_state.shape) if self.train_weights else None
        self.grad_b = np.zeros(self.bias.shape) if self.train_bias else None

        # Allocate memory for input, net, and output tensors
        self.input = np.zeros((batch_size, time_steps, input_size, 1))  # Store input sequences
        self.net = np.zeros((batch_size, time_steps, output_size, 1))   # Store pre-activation values
        self.output = np.zeros((batch_size, time_steps, output_size, 1))  # Store output (post-activation) values

    #################################################################

    def trainable_params(self) -> int:
        """
        Computes the total number of trainable parameters in the layer.

        Returns:
            int: Total number of trainable weights and biases.
        """
        params = 0
        if self.train_bias:  # Add bias parameters if trainable
            params += np.size(self.bias)
        if self.train_weights:  # Add weight parameters if trainable
            params += np.size(self.weight)
            params += np.size(self.weight_state)
        return params

    #################################################################

    def all_params(self) -> int:
        """
        Computes the total number of parameters (trainable and non-trainable).

        Returns:
            int: Total number of weights and biases in the layer.
        """
        params = np.size(self.weight) + np.size(self.weight_state)  # Total weight parameters
        if self.use_bias:  # Include bias parameters if applicable
            params += np.size(self.bias)
        return params

    #################################################################

    def __call__(self, batch_index: int, seq_index: int, input: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the layer.

        Parameters:
            batch_index (int): Index of the current batch.
            seq_index (int): Index of the current time step within the sequence.
            input (np.ndarray): Input data for the current time step.
            state (np.ndarray): Feedback state vector for the current time step.

        Returns:
            np.ndarray: Output of the layer for the current time step.
        """
        self.input[batch_index, seq_index] = input.reshape((-1, 1))  # Store the input

        # Validate the batch index
        if self.batch_size < batch_index:
            raise ValueError("Batch index exceeds the allowed batch size.")

        # Compute the net input using weights, state, and biases
        self.net[batch_index, seq_index] = (
            self.weight @ input.reshape((-1, 1)) +
            self.weight_state @ state.reshape((-1, 1))
        )
        if self.use_bias:  # Add biases if applicable
            self.net[batch_index, seq_index] += self.bias

        # Apply activation function
        self.output[batch_index, seq_index] = net2out(
            self.net[batch_index, seq_index], self.activation, alpha=self.alpha_activation
        )

        return self.output[batch_index, seq_index]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for the layer.

        Parameters:
            optimizer (str): Name of the optimization algorithm (e.g., 'Adam', 'SGD').
            **kwargs: Additional configuration parameters for the optimizer.
        """
        # Set up the optimizer for all trainable parameters
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, batch_size: int, learning_rate: float = 1e-3, grads: np.ndarray = None) -> None:
        """
        Updates weights and biases using gradients computed during backpropagation.

        Parameters:
            batch_size (int): Number of sequences in the current batch.
            learning_rate (float): Learning rate for parameter updates. Defaults to 1e-3.
            grads (np.ndarray, optional): Gradients of the parameters. If None, uses regularized gradients.
        """
        if grads is None:
            # Apply regularization to gradients if no external gradients are provided
            if self.train_weights:
                self.grad_w += (self.L1_coe * np.sign(self.weight) + self.L2_coe * self.weight)
                self.grad_w_state += (self.L1_coe * np.sign(self.weight_state) + self.L2_coe * self.weight_state)

            # Combine all gradients into a single array
            grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
            if grads is not None:
                if self.grad_w is not None:
                    grads = np.concatenate((grads, self.grad_w.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_state.reshape((-1, 1))))
                if self.grad_b is not None:
                    grads = np.concatenate((grads, self.grad_b.reshape((-1, 1))))
                grads /= batch_size  # Normalize gradients by batch size

        # Use the optimizer to compute updates for the parameters
        deltas = self.Optimizer(grads, learning_rate)

        # Apply updates to weights and biases
        ind2 = 0
        if self.train_weights:
            # Update main weights
            ind1 = ind2
            ind2 += int(np.size(self.weight))
            delta_w = deltas[ind1:ind2].reshape(self.weight.shape)
            self.weight -= delta_w  # Update weights

            # Update state weights
            ind1 = ind2
            ind2 += int(np.size(self.weight_state))
            delta_w_state = deltas[ind1:ind2].reshape(self.weight_state.shape)
            self.weight_state -= delta_w_state

        # Update biases if applicable
        if self.train_bias:
            ind1 = ind2
            ind2 += np.size(self.bias)
            delta_bias = deltas[ind1:ind2].reshape(self.bias.shape)
            self.bias -= delta_bias

        # Reset gradients for the next iteration
        self.grad_w = self.grad_w * 0 if self.train_weights else None
        self.grad_w_state = self.grad_w_state * 0 if self.train_weights else None
        self.grad_b = self.grad_b * 0 if self.train_bias else None

    #################################################################

    def return_grads(self) -> np.ndarray:
        """
        Retrieves the accumulated gradients for all trainable parameters.

        Returns:
            np.ndarray: A single array containing all gradients for weights and biases.
        """
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
        if self.grad_w is not None:
            grad_w = self.grad_w + (self.L1_coe * np.sign(self.weight) + self.L2_coe * self.weight)
            grad_w_state = self.grad_w_state + (self.L1_coe * np.sign(self.weight_state) + self.L2_coe * self.weight_state)
            grads = np.concatenate((grads, grad_w.reshape((-1, 1))))
            grads = np.concatenate((grads, grad_w_state.reshape((-1, 1))))
        if self.grad_b is not None:
            grads = np.concatenate((grads, self.grad_b.reshape((-1, 1))))
        return grads

    #################################################################

    def backward(self, batch_index: int, seq_index: int, error: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs backpropagation for a single sequence, computing gradients and propagating the error backward.

        Parameters:
            batch_index (int): Index of the current batch.
            seq_index (int): Index of the current time step within the sequence.
            error (np.ndarray): Error from the subsequent layer or time step.
            state (np.ndarray): Feedback state vector for the current time step.

        Returns:
            tuple[np.ndarray, np.ndarray]: Errors propagated to the input and state of the previous layer.
        """
        # Compute the derivative of the activation function
        Fprime = net2Fprime(self.net[batch_index, seq_index], self.activation, self.alpha_activation)

        # Compute sensitivity based on the activation function
        if self.activation == 'softmax':
            sensitivity = (Fprime @ error).reshape((-1, 1))
        else:
            sensitivity = error.reshape((-1, 1)) * Fprime

        # Accumulate gradients for weights and biases
        if self.train_weights:
            self.grad_w += np.outer(sensitivity.ravel(), self.input[batch_index, seq_index].ravel())  # Input gradients
            self.grad_w_state += np.outer(sensitivity.ravel(), state.ravel())  # State gradients
        if self.train_bias:
            self.grad_b += sensitivity

        # Compute error to propagate to the previous layer
        error_in = np.ravel(self.weight.T @ sensitivity)
        error_state = np.ravel(self.weight_state.T @ sensitivity)

        return error_in.reshape((-1, 1)), error_state.reshape((-1, 1))
    
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################
    ##################################################################################################################################

class Dense2Feedback:
    def __init__(self, time_steps: int, input_size: int, output_size: int, feedback_size_jordan: int,
                 use_bias: bool = True, train_bias: bool = True, train_weights: bool = True, batch_size: int = 32, 
                 activation: str = 'relu', alpha: float = None, weights_init_method: str = 'he', 
                 weight_distribution: str = 'normal', orthogonal_scale_factor: float = 1.0, 
                 weights_uniform_range: tuple = None, L2_coe: float = 0.0, L1_coe: float = 0.0):
        """
        Initializes a dense (fully connected) layer with dual feedback (Elman and Jordan networks).

        Parameters:
            time_steps (int): Number of time steps or sequences in the input data.
            input_size (int): Number of input features (size of the input vector).
            output_size (int): Number of output features (size of the output vector).
            feedback_size_jordan (int): Size of the Jordan feedback vector (state from previous layer).
            use_bias (bool): Whether to include bias terms in the layer. Defaults to True.
            train_bias (bool): Whether to allow training of bias parameters. Defaults to True.
            train_weights (bool): Whether to allow training of weight parameters. Defaults to True.
            batch_size (int): Number of sequences per training batch. Defaults to 32.
            activation (str): Name of the activation function to use (e.g., 'relu', 'sigmoid'). Defaults to 'relu'.
            alpha (float, optional): Parameter for activation functions like Leaky ReLU and ELU. Defaults to None.
            weights_init_method (str): Method for initializing weights ('he', 'xavier', etc.). Defaults to 'he'.
            weight_distribution (str): Distribution type for weights ('normal', 'uniform'). Defaults to 'normal'.
            orthogonal_scale_factor (float): Scaling factor for orthogonal initialization. Defaults to 1.0.
            weights_uniform_range (tuple, optional): Range for uniform weight initialization. Defaults to None.
            L2_coe (float): Coefficient for L2 regularization. Defaults to 0.0.
            L1_coe (float): Coefficient for L1 regularization. Defaults to 0.0.
        """
        self.output_size = output_size  # Number of neurons in the output layer
        self.input_size = input_size    # Number of input neurons
        self.feedback_size_elman = output_size  # Elman feedback size (equals output size)
        self.feedback_size_jordan = feedback_size_jordan  # Jordan feedback size
        self.batch_size = batch_size    # Number of sequences per batch
        self.time_steps = time_steps    # Number of time steps in each sequence
        self.use_bias = use_bias        # Whether to include bias terms
        self.train_bias = train_bias if use_bias else False  # Allow training of bias only if bias is used
        self.train_weights = train_weights  # Allow training of weights
        self.activation = activation    # Activation function name
        self.L2_coe = L2_coe            # L2 regularization coefficient
        self.L1_coe = L1_coe            # L1 regularization coefficient
        self.alpha_activation = alpha   # Optional alpha parameter for advanced activation functions

        # Initialize weights for input, Elman feedback, and Jordan feedback
        self.weight = Dense_weight_init(input_size, output_size, method=weights_init_method, 
                                        distribution=weight_distribution, scale_factor=orthogonal_scale_factor, 
                                        ranges=weights_uniform_range)
        self.weight_jordan = Dense_weight_init(feedback_size_jordan, output_size, method=weights_init_method, 
                                               distribution=weight_distribution, scale_factor=orthogonal_scale_factor, 
                                               ranges=weights_uniform_range)
        self.weight_elman = Dense_weight_init(output_size, output_size, method=weights_init_method, 
                                              distribution=weight_distribution, scale_factor=orthogonal_scale_factor, 
                                              ranges=weights_uniform_range)

        # Initialize biases if applicable
        if self.use_bias:
            self.bias = np.zeros((output_size, 1))  # Bias initialized to zero

        # Gradients for weights and biases
        self.grad_w = np.zeros(self.weight.shape) if self.train_weights else None
        self.grad_w_elman = np.zeros(self.weight_elman.shape) if self.train_weights else None
        self.grad_w_jordan = np.zeros(self.weight_jordan.shape) if self.train_weights else None
        self.grad_b = np.zeros(self.bias.shape) if self.train_bias else None

        # Allocate memory for input, net, and output tensors
        self.input = np.zeros((batch_size, time_steps, input_size, 1))  # Store input sequences
        self.net = np.zeros((batch_size, time_steps, output_size, 1))   # Store pre-activation values
        self.output = np.zeros((batch_size, time_steps, output_size, 1))  # Store output (post-activation) values

    #################################################################

    def trainable_params(self) -> int:
        """
        Computes the total number of trainable parameters in the layer.

        Returns:
            int: Total number of trainable weights and biases.
        """
        params = 0
        if self.train_bias:  # Count trainable biases
            params += np.size(self.bias)
        if self.train_weights:  # Count trainable weights
            params += np.size(self.weight)
            params += np.size(self.weight_elman)
            params += np.size(self.weight_jordan)
        return params

    #################################################################

    def all_params(self) -> int:
        """
        Computes the total number of parameters (trainable and non-trainable).

        Returns:
            int: Total number of weights and biases in the layer.
        """
        params = np.size(self.weight) + np.size(self.weight_elman) + np.size(self.weight_jordan)  # Count all weights
        if self.use_bias:  # Include biases if applicable
            params += np.size(self.bias)
        return params

    #################################################################

    def __call__(self, batch_index: int, seq_index: int, input: np.ndarray,
                 elman_state: np.ndarray, jordan_state: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the layer.

        Parameters:
            batch_index (int): Index of the current batch.
            seq_index (int): Index of the current time step within the sequence.
            input (np.ndarray): Input data for the current time step.
            elman_state (np.ndarray): Feedback state vector from the layer's own output (Elman feedback).
            jordan_state (np.ndarray): Feedback state vector from the previous layer (Jordan feedback).

        Returns:
            np.ndarray: Output of the layer for the current time step.
        """
        self.input[batch_index, seq_index] = input.reshape((-1, 1))  # Store the input

        # Validate the batch index
        if self.batch_size < batch_index:
            raise ValueError("Batch index exceeds the allowed batch size.")

        # Compute the net input using weights, feedback, and biases
        self.net[batch_index, seq_index] = (
            self.weight @ input.reshape((-1, 1)) +
            self.weight_elman @ elman_state.reshape((-1, 1)) +
            self.weight_jordan @ jordan_state.reshape((-1, 1))
        )
        if self.use_bias:  # Add biases if applicable
            self.net[batch_index, seq_index] += self.bias

        # Apply activation function
        self.output[batch_index, seq_index] = net2out(
            self.net[batch_index, seq_index], self.activation, alpha=self.alpha_activation
        )

        return self.output[batch_index, seq_index]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for the layer using the specified method.

        Parameters:
            optimizer (str): Name of the optimization algorithm (e.g., 'Adam', 'SGD'). Defaults to 'Adam'.
            **kwargs: Additional configuration parameters for the optimizer.
        """
        # Initialize the optimizer for all trainable parameters in the layer
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, batch_size: int, learning_rate: float = 1e-3, grads: np.ndarray = None) -> None:
        """
        Updates the layer's weights and biases based on calculated gradients.

        Parameters:
            batch_size (int): Number of sequences in the current batch.
            learning_rate (float): Learning rate for parameter updates. Defaults to 1e-3.
            grads (np.ndarray, optional): Gradients of the parameters. If None, uses internally computed gradients.

        Returns:
            None
        """
        if grads is None:
            # Apply regularization to gradients if no external gradients are provided
            if self.train_weights:
                self.grad_w += (self.L1_coe * np.sign(self.weight) + self.L2_coe * self.weight)
                self.grad_w_elman += (self.L1_coe * np.sign(self.weight_elman) + self.L2_coe * self.weight_elman)
                self.grad_w_jordan += (self.L1_coe * np.sign(self.weight_jordan) + self.L2_coe * self.weight_jordan)

            # Combine all gradients into a single array
            grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
            if grads is not None:
                if self.grad_w is not None:
                    grads = np.concatenate((grads, self.grad_w.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_elman.reshape((-1, 1))))
                    grads = np.concatenate((grads, self.grad_w_jordan.reshape((-1, 1))))
                if self.grad_b is not None:
                    grads = np.concatenate((grads, self.grad_b.reshape((-1, 1))))
                grads /= batch_size  # Normalize gradients by batch size

        # Use the optimizer to compute updates for the parameters
        deltas = self.Optimizer(grads, learning_rate)

        # Apply updates to weights and biases
        ind2 = 0
        if self.train_weights:
            # Update main weights
            ind1 = ind2
            ind2 += int(np.size(self.weight))
            delta_w = deltas[ind1:ind2].reshape(self.weight.shape)
            self.weight -= delta_w  # Apply weight update

            # Update Elman feedback weights
            ind1 = ind2
            ind2 += int(np.size(self.weight_elman))
            delta_w_elman = deltas[ind1:ind2].reshape(self.weight_elman.shape)
            self.weight_elman -= delta_w_elman  # Apply weight update

            # Update Jordan feedback weights
            ind1 = ind2
            ind2 += int(np.size(self.weight_jordan))
            delta_w_jordan = deltas[ind1:ind2].reshape(self.weight_jordan.shape)
            self.weight_jordan -= delta_w_jordan  # Apply weight update

        # Update biases if applicable
        if self.train_bias:
            ind1 = ind2
            ind2 += np.size(self.bias)
            delta_bias = deltas[ind1:ind2].reshape(self.bias.shape)
            self.bias -= delta_bias  # Apply bias update

        # Reset gradients for the next iteration
        self.grad_w = self.grad_w * 0 if self.train_weights else None
        self.grad_w_elman = self.grad_w_elman * 0 if self.train_weights else None
        self.grad_w_jordan = self.grad_w_jordan * 0 if self.train_weights else None
        self.grad_b = self.grad_b * 0 if self.train_bias else None

    #################################################################

    def return_grads(self) -> np.ndarray:
        """
        Retrieves the accumulated gradients for all trainable parameters.

        Returns:
            np.ndarray: A single array containing all gradients for weights and biases.
        """
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
        if self.grad_w is not None:
            grad_w = self.grad_w + (self.L1_coe * np.sign(self.weight) + self.L2_coe * self.weight)
            grad_w_elman = self.grad_w_elman + (self.L1_coe * np.sign(self.weight_elman) + self.L2_coe * self.weight_elman)
            grad_w_jordan = self.grad_w_jordan + (self.L1_coe * np.sign(self.weight_jordan) + self.L2_coe * self.weight_jordan)
            grads = np.concatenate((grads, grad_w.reshape((-1, 1))))
            grads = np.concatenate((grads, grad_w_elman.reshape((-1, 1))))
            grads = np.concatenate((grads, grad_w_jordan.reshape((-1, 1))))
        if self.grad_b is not None:
            grads = np.concatenate((grads, self.grad_b.reshape((-1, 1))))
        return grads

    #################################################################

    def backward(self, batch_index: int, seq_index: int, error: np.ndarray,
                 elman_state: np.ndarray, jordan_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs backpropagation for a single sequence, computing gradients and propagating the error backward.

        Parameters:
            batch_index (int): Index of the current batch.
            seq_index (int): Index of the current time step within the sequence.
            error (np.ndarray): Error from the subsequent layer or time step.
            elman_state (np.ndarray): Feedback state vector from the layer's own output (Elman feedback).
            jordan_state (np.ndarray): Feedback state vector from the previous layer (Jordan feedback).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Errors propagated to the input, Elman, and Jordan feedback.
        """
        # Compute the derivative of the activation function
        Fprime = net2Fprime(self.net[batch_index, seq_index], self.activation, self.alpha_activation)

        # Compute sensitivity based on the activation function
        if self.activation == 'softmax':
            sensitivity = (Fprime @ error).reshape((-1, 1))
        else:
            sensitivity = error.reshape((-1, 1)) * Fprime

        # Accumulate gradients for weights and biases
        if self.train_weights:
            self.grad_w += np.outer(sensitivity.ravel(), self.input[batch_index, seq_index].ravel())  # Input gradients
            self.grad_w_elman += np.outer(sensitivity.ravel(), elman_state.ravel())  # Elman feedback gradients
            self.grad_w_jordan += np.outer(sensitivity.ravel(), jordan_state.ravel())  # Jordan feedback gradients
        if self.train_bias:
            self.grad_b += sensitivity

        # Compute error to propagate to the previous layer
        error_in = np.ravel(self.weight.T @ sensitivity)
        error_elman = np.ravel(self.weight_elman.T @ sensitivity)
        error_jordan = np.ravel(self.weight_jordan.T @ sensitivity)

        return error_in.reshape((-1, 1)), error_elman.reshape((-1, 1)), error_jordan.reshape((-1, 1))
