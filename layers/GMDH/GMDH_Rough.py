import numpy as np
from itertools import combinations
from initializers.weight_initializer import Dense_weight_init
from optimizers.set_optimizer import init_optimizer

class GMDHRough:
    """
    A GMDH (Group Method of Data Handling) Rough Model implementation.

    Attributes:
        input_size (int): The size of the input feature set.
        output_size (int): The number of output nodes determined by input combinations.
        batch_size (int): The size of data batches.
        use_bias (bool): Whether to use bias terms.
        train_bias (bool): Whether to train the bias terms.
        train_weights (bool): Whether to train the weights.
        train_blending (bool): Whether to train blending factors.
        activation (str): Placeholder for activation function (currently not used).
        L2_coe (float): L2 regularization coefficient.
        L1_coe (float): L1 regularization coefficient.
        upper_weight (np.ndarray): Upper layer weights initialized with Dense_weight_init.
        lower_weight (np.ndarray): Lower layer weights initialized with Dense_weight_init.
        upper_bias (np.ndarray): Upper layer bias terms.
        lower_bias (np.ndarray): Lower layer bias terms.
        blending_factor (np.ndarray): Factors for blending upper and lower outputs.
        minmax_reverse_stat (np.ndarray): Stores reverse min-max statistics for each batch.
        output (np.ndarray): Stores the final output of the model.
        lower_output (np.ndarray): Stores the lower layer output.
        upper_output (np.ndarray): Stores the upper layer output.
        input_inds (np.ndarray): Stores index combinations of input features.
    """

    def __init__(
        self, 
        input_size: int, 
        use_bias: bool = True, 
        train_bias: bool = True,
        train_weights: bool = True, 
        train_blending: bool = False, 
        batch_size: int = 32,
        weights_uniform_range: tuple = (-1, 1), 
        L2_coe: float = 0.0, 
        L1_coe: float = 0.0
    ) -> None:
        """
        Initializes the GMDHRough model.

        Args:
            input_size (int): Number of input features.
            use_bias (bool): Whether to use bias terms.
            train_bias (bool): Whether to allow training of bias terms.
            train_weights (bool): Whether to allow training of weights.
            train_blending (bool): Whether to allow training of blending factors.
            batch_size (int): Batch size for processing.
            weights_uniform_range (tuple): Range for uniform weight initialization.
            L2_coe (float): L2 regularization coefficient.
            L1_coe (float): L1 regularization coefficient.

        Raises:
            ValueError: If input_size is less than 2.
        """
        if input_size < 2:
            raise ValueError('GMDH input size cannot be less than 2')
        self.input_size = input_size
        self.output_size = np.math.factorial(input_size) // (np.math.factorial(2) * np.math.factorial(input_size - 2))
        self.batch_size = batch_size
        self.use_bias = use_bias
        self.train_bias = False if use_bias is False else train_bias
        self.train_weights = train_weights
        self.train_blending = train_blending
        self.activation = 'None'
        self.L2_coe = L2_coe
        self.L1_coe = L1_coe
        middle = (weights_uniform_range[0] + weights_uniform_range[1]) / 2
        upper_range = (weights_uniform_range[0], middle)
        lower_range = (middle, weights_uniform_range[1])
        
        # Initialize weights using Dense_weight_init
        self.upper_weight = Dense_weight_init(5, self.output_size, method="uniform", ranges=upper_range)
        self.lower_weight = Dense_weight_init(5, self.output_size, method="uniform", ranges=lower_range)
        
        # Initialize bias if use_bias is True
        if self.use_bias:
            self.upper_bias = np.zeros((self.output_size, 1))
            self.lower_bias = np.zeros((self.output_size, 1))
        
        # Initialize blending factors, outputs, and input combinations
        self.blending_factor = np.zeros((self.output_size, 1)) + 0.5
        self.minmax_reverse_stat = np.zeros((batch_size, self.output_size, 1))
        self.output = np.zeros((batch_size, self.output_size, 1))
        self.lower_output = np.zeros((batch_size, self.output_size, 1))
        self.upper_output = np.zeros((batch_size, self.output_size, 1))
        self.input_inds = np.array(list(combinations(range(input_size), 2)))

    #################################################################

    def trainable_params(self) -> int:
        """
        Calculates the number of trainable parameters in the model.

        Returns:
            int: Total number of trainable parameters.
        """
        params = 0
        if self.train_bias:
            params += np.size(self.upper_bias) * 2
        if self.train_weights:
            params += np.size(self.upper_weight) * 2
        if self.train_blending:
            params += np.size(self.blending_factor)
        return params

    def all_params(self) -> int:
        """
        Calculates the total number of parameters in the model (trainable and non-trainable).

        Returns:
            int: Total number of parameters.
        """
        params = np.size(self.upper_weight) * 2 + np.size(self.blending_factor)
        if self.use_bias:
            params += np.size(self.upper_bias) * 2
        return params
    
    #################################################################

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Performs forward propagation through the model.

        Args:
            input (np.ndarray): Input data of shape (batch_size, input_size).

        Returns:
            np.ndarray: Model output of shape (batch_size, output_size).
        """
        self.input = input
        if self.batch_size < input.shape[0]:
            raise ValueError('Data batch size cannot be larger than model batch size')
        for batch_index, input_vector in enumerate(input):
            adalines_inputs = input_vector[self.input_inds]
            
            # Compute outputs for upper and lower weights
            up_out = (
                adalines_inputs[:, 0] * self.upper_weight[:, 0] +
                np.square(adalines_inputs[:, 0]) * self.upper_weight[:, 1] +
                adalines_inputs[:, 0] * adalines_inputs[:, 1] * self.upper_weight[:, 2] +
                np.square(adalines_inputs[:, 1]) * self.upper_weight[:, 3] +
                adalines_inputs[:, 1] * self.upper_weight[:, 4]
            ).reshape((-1, 1))
            
            low_out = (
                adalines_inputs[:, 0] * self.lower_weight[:, 0] +
                np.square(adalines_inputs[:, 0]) * self.lower_weight[:, 1] +
                adalines_inputs[:, 0] * adalines_inputs[:, 1] * self.lower_weight[:, 2] +
                np.square(adalines_inputs[:, 1]) * self.lower_weight[:, 3] +
                adalines_inputs[:, 1] * self.lower_weight[:, 4]
            ).reshape((-1, 1))
            
            # Apply bias if enabled
            if self.use_bias:
                self.upper_output[batch_index] += self.upper_bias
                self.lower_output[batch_index] += self.lower_bias
            
            # Concatenate outputs and compute min/max statistics
            concat_out = np.concatenate((up_out, low_out), axis=1)
            self.minmax_reverse_stat[batch_index] = np.argmax(concat_out).reshape(-1, 1)
            self.upper_output[batch_index] = np.max(concat_out, axis=1).reshape((-1, 1))
            self.lower_output[batch_index] = np.min(concat_out, axis=1).reshape((-1, 1))
            
            # Blend outputs based on blending factor
            self.output[batch_index] = self.blending_factor * self.upper_output[batch_index] + \
                (1 - self.blending_factor) * self.lower_output[batch_index]

        batch_index += 1
        return self.output[:batch_index, :, 0]

    #################################################################

    def LMS(self, input: np.ndarray, output: np.ndarray) -> None:
        """
        Performs the Least Mean Squares (LMS) regression to update weights and biases for the model.

        Args:
            input (np.ndarray): Input data of shape (batch_size, input_size).
            output (np.ndarray): Target output data of shape (batch_size, output_size).

        Raises:
            ValueError: If input/output shapes are incompatible or batch sizes do not match.
        """
        # Validate input dimensions
        if np.ndim(input) != 2 or input.shape[1] != self.input_size:
            raise ValueError("Input must be of shape (batch_size, input_size).")
        if np.ndim(output) != 2 or output.shape[1] != self.output_size:
            raise ValueError("Output must be of shape (batch_size, output_size).")
        if input.shape[0] != output.shape[0]:
            raise ValueError("Input and output batch sizes must match.")

        # Iterate through each output node for LMS computation
        for i in range(self.output_size):
            index = self.input_inds[i]  # Get input indices for the current combination
            z1 = input[:, index[0]].reshape((-1, 1))  # First input feature in the pair
            z2 = input[:, index[1]].reshape((-1, 1))  # Second input feature in the pair
            Y = output[:, i].reshape((-1, 1))  # Target output for the current node

            # Compute blending factors alpha and beta
            alpha = self.blending_factor[i]
            beta = 1 - self.blending_factor[i]

            # Construct the design matrix X with blended inputs
            X = np.concatenate((
                alpha * z1,                      # Linear term (z1)
                alpha * z1 ** 2,                 # Quadratic term (z1^2)
                alpha * z1 * z2,                 # Interaction term (z1 * z2)
                alpha * z2 ** 2,                 # Quadratic term (z2^2)
                alpha * z2,                      # Linear term (z2)
                alpha * np.ones(z1.shape),       # Bias term
                beta * z1,                       # linear term (z1)
                beta * z1 ** 2,                  # quadratic term (z1^2)
                beta * z1 * z2,                  # interaction term (z1 * z2)
                beta * z2 ** 2,                  # quadratic term (z2^2)
                beta * z2,                       # linear term (z2)
                beta * np.ones(z1.shape)         # bias term
            ), axis=1)

            # Solve for weights using the normal equation
            W = np.ravel(np.linalg.inv(X.T @ X) @ X.T @ Y)

            # Split weights into upper and lower components
            W_up = W[:6]
            W_low = W[6:]

            # Assign weights and biases to the respective layers
            self.upper_weight[i] = W_up[:-1]  # Exclude bias from weights
            self.lower_weight[i] = W_low[:-1]
            self.upper_bias[i] = W_up[-1]  # Assign last element as bias
            self.lower_bias[i] = W_low[-1]

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initializes the optimizer for updating trainable parameters.

        Args:
            optimizer (str): The optimization algorithm to use (default: 'Adam').
            **kwargs: Additional parameters for the optimizer.
        """
        # Initialize the optimizer with trainable parameter count and provided arguments
        self.Optimizer = init_optimizer(self.trainable_params(), method=optimizer, **kwargs)

    #################################################################

    def update(self, grads: np.ndarray, learning_rate: float = 1e-3) -> None:
        """
        Updates model parameters using computed gradients.

        Args:
            grads (np.ndarray): Gradients for all trainable parameters.
            learning_rate (float): Learning rate for parameter updates (default: 1e-3).
        """
        # Apply optimizer to get parameter deltas
        deltas = self.Optimizer(grads, learning_rate)

        ind2 = 0  # Index tracker for gradients
        if self.train_weights:
            # Update upper layer weights
            ind1 = ind2
            ind2 += int(np.size(self.upper_weight))
            delta_w = deltas[ind1:ind2].reshape(self.upper_weight.shape)
            self.upper_weight -= delta_w

            # Update lower layer weights
            ind1 = ind2
            ind2 += int(np.size(self.lower_weight))
            delta_w = deltas[ind1:ind2].reshape(self.lower_weight.shape)
            self.lower_weight -= delta_w

        if self.train_bias:
            # Update upper layer bias
            ind1 = ind2
            ind2 += np.size(self.upper_bias)
            delta_bias = deltas[ind1:ind2].reshape(self.upper_bias.shape)
            self.upper_bias -= delta_bias

            # Update lower layer bias
            ind1 = ind2
            ind2 += np.size(self.lower_bias)
            delta_bias = deltas[ind1:ind2].reshape(self.lower_bias.shape)
            self.lower_bias -= delta_bias

        if self.train_blending:
            # Update blending factors
            ind1 = ind2
            ind2 += np.size(self.blending_factor)
            delta_b = deltas[ind1:ind2].reshape(self.blending_factor.shape)
            self.blending_factor -= delta_b

    #################################################################

    def backward(
        self,
        error_batch: np.ndarray,
        learning_rate: float = 1e-3,
        return_error: bool = False,
        return_grads: bool = False,
        modify: bool = True
    ):
        """
        Perform the backward pass to compute gradients and update parameters.

        Args:
            error_batch (np.ndarray): The batch of errors with shape (batch_size, output_size).
            learning_rate (float, optional): The learning rate for updating parameters. Default is 1e-3.
            return_error (bool, optional): Whether to return the propagated input error. Default is False.
            return_grads (bool, optional): Whether to return the computed gradients. Default is False.
            modify (bool, optional): Whether to update the model's parameters. Default is True.

        Returns:
            Union[None, dict[str, np.ndarray], np.ndarray]:
                - None if `modify` is True and no gradients or errors are requested.
                - A dictionary containing the propagated error and gradients if both `return_error` and `return_grads` are True.
                - The propagated error if `return_error` is True.
                - The computed gradients if `return_grads` is True.
        """
        # Initialize the error propagation array if required
        if return_error:
            error_in = np.zeros(self.input.shape)  # Shape matches the input tensor

        # Initialize gradient storage variables if the respective parameters are trainable
        grad_w_up = np.zeros(self.upper_weight.shape) if self.train_weights else None
        grad_w_low = np.zeros(self.lower_weight.shape) if self.train_weights else None
        grad_bias_up = np.zeros(self.upper_bias.shape) if self.train_bias else None
        grad_bias_low = np.zeros(self.lower_bias.shape) if self.train_bias else None
        grad_blending = np.zeros(self.blending_factor.shape) if self.train_blending else None

        # Process each batch in the error batch
        for batch_index, one_batch_error in enumerate(error_batch):
            input_vector = self.input[batch_index]  # Input corresponding to the batch
            adalines_inputs = input_vector[self.input_inds]  # Extract inputs based on indices
            one_batch_error = one_batch_error.reshape((-1, 1))  # Ensure the error is columnar

            # Compute blending factor gradients if enabled
            if self.train_blending:
                grad_alpha += one_batch_error * (
                    self.upper_output[batch_index] - self.lower_output[batch_index]
                )

            # Separate error into upper and lower components
            e_max = self.blending_factor * one_batch_error
            e_min = (1 - self.blending_factor) * one_batch_error
            e_upper = (
                e_max * np.logical_not(self.minmax_reverse_stat[batch_index])
                + e_min * self.minmax_reverse_stat[batch_index]
            ).ravel()
            e_lower = (
                e_min * np.logical_not(self.minmax_reverse_stat[batch_index])
                + e_max * self.minmax_reverse_stat[batch_index]
            ).ravel()

            # Compute weight gradients if trainable
            if self.train_weights:
                grad_w1_up = adalines_inputs[:, 0] * e_upper  # Gradient for w1 in upper weights
                grad_w5_up = adalines_inputs[:, 1] * e_upper  # Gradient for w5 in upper weights
                grad_w2_up = np.square(adalines_inputs[:, 0]) * e_upper  # Gradient for w2
                grad_w4_up = np.square(adalines_inputs[:, 1]) * e_upper  # Gradient for w4
                grad_w3_up = adalines_inputs[:, 0] * adalines_inputs[:, 1] * e_upper  # Gradient for w3
                grad_w_up += np.concatenate(  # Combine all upper weight gradients
                    (
                        grad_w1_up.reshape((-1, 1)),
                        grad_w2_up.reshape((-1, 1)),
                        grad_w3_up.reshape((-1, 1)),
                        grad_w4_up.reshape((-1, 1)),
                        grad_w5_up.reshape((-1, 1)),
                    ),
                    axis=1,
                )

                grad_w1_low = adalines_inputs[:, 0] * e_lower  # Gradient for w1 in lower weights
                grad_w5_low = adalines_inputs[:, 1] * e_lower  # Gradient for w5 in lower weights
                grad_w2_low = np.square(adalines_inputs[:, 0]) * e_lower  # Gradient for w2
                grad_w4_low = np.square(adalines_inputs[:, 1]) * e_lower  # Gradient for w4
                grad_w3_low = adalines_inputs[:, 0] * adalines_inputs[:, 1] * e_lower  # Gradient for w3
                grad_w_low += np.concatenate(  # Combine all lower weight gradients
                    (
                        grad_w1_low.reshape((-1, 1)),
                        grad_w2_low.reshape((-1, 1)),
                        grad_w3_low.reshape((-1, 1)),
                        grad_w4_low.reshape((-1, 1)),
                        grad_w5_low.reshape((-1, 1)),
                    ),
                    axis=1,
                )

            # Compute bias gradients if trainable
            if self.train_bias:
                grad_bias_up += e_upper.reshape((-1, 1))  # Gradient for upper biases
                grad_bias_low += e_lower.reshape((-1, 1))  # Gradient for lower biases

            # Propagate error through the network if required
            if return_error:
                grad_x1_up = (
                    self.upper_weight[:, 0]
                    + 2 * self.upper_weight[:, 1] * adalines_inputs[:, 0]
                    + self.upper_weight[:, 2] * adalines_inputs[:, 1]
                ) * e_upper
                grad_x2_up = (
                    self.upper_weight[:, 4]
                    + 2 * self.upper_weight[:, 3] * adalines_inputs[:, 1]
                    + self.upper_weight[:, 2] * adalines_inputs[:, 0]
                ) * e_upper
                grad_x_up = np.concatenate(
                    (grad_x1_up.reshape((-1, 1)), grad_x2_up.reshape((-1, 1))), axis=1
                )

                # Calculate upper propagated error
                error_in_batch_up = np.zeros(self.input_size)
                np.add.at(error_in_batch_up, self.input_inds.ravel(), grad_x_up.ravel())

                # Repeat for lower weights
                grad_x1_low = (
                    self.lower_weight[:, 0]
                    + 2 * self.lower_weight[:, 1] * adalines_inputs[:, 0]
                    + self.lower_weight[:, 2] * adalines_inputs[:, 1]
                ) * e_lower
                grad_x2_low = (
                    self.lower_weight[:, 4]
                    + 2 * self.lower_weight[:, 3] * adalines_inputs[:, 1]
                    + self.lower_weight[:, 2] * adalines_inputs[:, 0]
                ) * e_lower
                grad_x_low = np.concatenate(
                    (grad_x1_low.reshape((-1, 1)), grad_x2_low.reshape((-1, 1))), axis=1
                )

                # Calculate lower propagated error
                error_in_batch_low = np.zeros(self.input_size)
                np.add.at(error_in_batch_low, self.input_inds.ravel(), grad_x_low.ravel())

                # Combine upper and lower propagated errors
                error_in[batch_index] = error_in_batch_up + error_in_batch_low

        # Normalize and regularize gradients for weights if applicable
        if self.train_weights:
            grad_w_up /= error_batch.shape[0]
            grad_w_up += self.L1_coe * np.sign(self.upper_weight) + self.L2_coe * self.upper_weight
            grad_w_low /= error_batch.shape[0]
            grad_w_low += self.L1_coe * np.sign(self.lower_weight) + self.L2_coe * self.lower_weight

        # Normalize bias gradients if applicable
        if self.train_bias:
            grad_bias_up /= error_batch.shape[0]
            grad_bias_low /= error_batch.shape[0]

        # Prepare gradient array for updating parameters
        grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
        if grads is not None:
            if self.train_weights:
                grads = np.concatenate((grads, grad_w_up.reshape((-1, 1))))
                grads = np.concatenate((grads, grad_w_low.reshape((-1, 1))))
            if self.train_bias:
                grads = np.concatenate((grads, grad_bias_up.reshape((-1, 1))))
                grads = np.concatenate((grads, grad_bias_low.reshape((-1, 1))))
            if self.train_blending:
                grads = np.concatenate((grads, grad_blending.reshape((-1, 1))))

        # Update parameters if `modify` is True
        if modify:
            self.update(grads, learning_rate=learning_rate)

        # Return requested outputs
        if return_error and return_grads:
            return {"error_in": error_in, "gradients": grads}
        elif return_error:
            return error_in
        elif return_grads:
            return grads
