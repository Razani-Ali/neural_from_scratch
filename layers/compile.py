import numpy as np
from visualizations.plot_metrics import plot_metrics
from IPython.display import clear_output



#############################################################################################################################

class compile:
    """
    Compile and manage the training and inference process of a neural network model.

    Parameters:
    -----------
    model : list
        A list of layers (objects) that define the structure of the neural network. Each layer should implement 
        a forward and backward method for inference and backpropagation, respectively.

    Attributes:
    -----------
    model : list
        The list of layers comprising the neural network.
    """

    def __init__(self, model: list):
        """
        Initialize the compilation class with the given model.

        Parameters:
        -----------
        model : list
            A list containing the layers of the model in the order of forward propagation.
        """
        # Validate layer compatibility: Check if each layer's output size matches the next layer's input size
        for i in range(len(model) - 1):
            if model[i].output_size != model[i + 1].input_size:
                raise ValueError(
                    f"Layer mismatch detected: Layer {i + 1} output size ({model[i].output_size}) "
                    f"does not match Layer {i + 2} input size ({model[i + 1].input_size})."
                )
        self.batch_size = min(layer.batch_size for layer in model)
        self.model = model

    #################################################################

    def trainable_params(self) -> int:
        """
        Calculates the total number of trainable parameters in the compiled model.

        This method iterates over all the layers in the model and sums up the number 
        of trainable parameters from each layer.

        Returns:
        int: The total number of trainable parameters in the model.
        """
    
        # Initialize a variable to keep track of the total number of parameters.
        params = 0

        # Iterate through each layer in the model
        for layer in self.model:
            # Add the number of trainable parameters from each layer
            params += layer.trainable_params()
        
        # Return the total count of trainable parameters
        return int(params)
    
    #################################################################

    def all_params(self) -> int:
        """
        Calculates the total number of parameters in the compiled model.

        This method iterates over all the layers in the model and sums up the number 
        of parameters from each layer.

        Returns:
        int: The total number of parameters in the model.
        """
    
        # Initialize a variable to keep track of the total number of parameters.
        params = 0

        # Iterate through each layer in the model
        for layer in self.model:
            # Add the number of parameters from each layer
            params += layer.all_params()
        
        # Return the total count of trainable parameters
        return int(params)
    
    #################################################################

    def summary(self) -> None:
        """
        Prints a detailed summary of the model's architecture, including the number of parameters (trainable
        and non-trainable) for each layer, along with activation functions and layer types.
        
        Returns
        -------
        None
            This method only outputs the model summary to the console.
        """
        # Print the title of the model summary with decorative asterisks
        print('\n', '*' * 30, 'model summary', '*' * 30, '\n')
        
        # Initialize counters for the total number of trainable and all parameters in the model
        total_n_trainable = 0
        total_n_all = 0
        
        # Iterate through each layer in the model and gather information for summary
        for index, layer in enumerate(self.model):
            # Print layer index (1-based) and type of layer
            print(f'layer {index+1}:', end='\n\t')
            print(type(layer), end='\n\t')
            
            # Print the activation function used in the current layer
            print('activation function:', layer.activation, end='\n\t')
            # Print batch size of model, input size and neuron numbers
            print('batch size:', layer.batch_size, end='\n\t')
            print('input size:', layer.input_size, end='\n\t')
            print('output size:', layer.output_size, end='\n\t')
            
            # Get the number of trainable parameters for the current layer
            n_trainable = layer.trainable_params()
            # Accumulate the trainable parameter count to the total
            total_n_trainable += n_trainable
            
            # Get the total number of parameters (trainable + non-trainable) in the current layer
            n_all = layer.all_params()
            # Accumulate the total parameter count
            total_n_all += n_all
            
            # Print the total number of parameters in the current layer
            print(f'number of parameters: {n_all}', end='\n\t')
            
            # Print the number of trainable parameters in the current layer
            print(f'number of trainable parameters: {n_trainable}', end='\n\t')
            
            # Calculate and print the number of non-trainable parameters in the current layer
            print(f'number of non trainable parameters: {n_all - n_trainable}', end='\n\t')
            
            # Print a separator line for clarity between layers
            print('-' * 50)
        
        # Print the total number of parameters across all layers in the model
        print(f'total number of parameters: {total_n_all}', end='\n\t')
        
        # Print the total number of trainable parameters across the model
        print(f'total number of trainable parameters: {total_n_trainable}', end='\n\t')
        
        # Print the total number of non-trainable parameters across the model
        print(f'total number of non trainable parameters: {total_n_all - total_n_trainable}', end='\n\t')

    #################################################################

    def validate_final_output(self, output: np.ndarray) -> None:
        """
        Validates the output of the final layer to ensure it matches the expected output size and dimensions.

        Parameters:
        -----------
        output : np.ndarray
            The output array from the final layer to validate.

        Raises:
        -------
        ValueError:
            If the output size does not match the expected output size of the last layer.
            If the output dimension is less than 2.
        """
        # Ensure the output has at least 2 dimensions (batch size and output size)
        if output.ndim < 2:
            raise ValueError(
                f"Invalid output dimensions: The final layer's output must have at least 2 dimensions, "
                f"but got {output.ndim} dimensions. Ensure the output includes batch size and output size."
            )

        # Retrieve the expected output size of the last layer
        final_layer_output_size = self.model[-1].output_size

        # Determine the expected output shape, handling both int and tuple cases
        if isinstance(final_layer_output_size, int):
            expected_output_shape = (final_layer_output_size,)
        else:
            expected_output_shape = final_layer_output_size

        # Validate that the provided output size matches the expected output size of the last layer
        if output.shape[1:] != expected_output_shape:
            raise ValueError(
                f"Final output size mismatch: Model produced output size {output.shape[1:]}, "
                f"which does not match the expected output size {expected_output_shape} "
                "of the final layer."
            )

    #################################################################

    def backward(self, input: np.ndarray, targets: np.ndarray, Loss_function, batch_size: int = 8,
                 learning_rate: float = 1e-3, shuffle: bool = True) -> None:
        """
        Perform the backpropagation step to update the model's weights.

        Parameters:
        -----------
        input : np.ndarray
            Input data for the current batch.
        targets : np.ndarray
            The ground truth labels corresponding to the input data.
        Loss_function : object
            The loss function object that computes both the forward and backward pass of the loss.
        batch_size : int, optional
            Training batch_size
        learning_rate : float, optional
            Learning rate for updating weights (default is 1e-3).
        shuffle : bool, optionaal
            Shuffle data before training process
        """
        # Validate input size: Ensure input size matches the first layer's input size (ignoring batch size)
        self.validate_input(input)
        self.validate_final_output(targets)

        # Shuffle data if needed
        if shuffle:
            random_indices = np.random.permutation(input.shape[0])
            input = input[random_indices]
            targets = targets[random_indices]
        # Determine how many batches are needed based on batch size
        batch_num = int(np.ceil(input.shape[0] // batch_size))

        # Iterate over each batch
        for i in range(batch_num):
            # Extract the current batch of input data and corresponding targets
            data_X = input[i * batch_size: (i + 1) * batch_size].copy()
            data_Y = targets[i * batch_size: (i + 1) * batch_size].copy()

            # Perform forward pass to get the output
            out = self(data_X)
            
            # Calculate the error using the loss function's backward method
            if hasattr(Loss_function, 'memory'):
                _ = Loss_function.forward(out, data_Y)
                error = Loss_function.backward()
            else:
                error = Loss_function.backward(out, data_Y)

            # Perform backpropagation on each layer, starting from the last layer
            for layer in reversed(self.model):
                if layer == self.model[0]:
                    layer.backward(error, learning_rate=learning_rate)
                else:
                    error = layer.backward(error, learning_rate=learning_rate, return_error=True)

    #################################################################

    def fit(self, X_train: np.ndarray, X_val: np.ndarray, Y_train: np.ndarray, Y_val: np.ndarray, Loss_function,
            batch_size: int = 8, epoch: int = 15, method: str = 'Adam', learning_rate: float = 1e-3, **kwargs) -> dict:
        """
        Train the model using the provided training and validation data.

        Parameters:
        -----------
        X_train : np.ndarray
            Training input data.
        X_val : np.ndarray
            Validation input data.
        Y_train : np.ndarray
            Training ground truth labels.
        Y_val : np.ndarray
            Validation ground truth labels.
        Loss_function : object
            The loss function object used to compute loss and gradients.
        batch_size : int, optional
            Training batch_size
        epoch : int, optional
            Number of epochs to train (default is 15).
        method : str, optional
            Optimization method (default is 'Adam').
        learning_rate : float, optional
            Learning rate for weight updates (default is 1e-3).   
        *kwargs: Additional plotting options:
            - plot_loss (bool): Whether to plot loss curves (default: True).
            - plot_fitting (bool): Whether to plot fitting for regression tasks (default: False).
            - plot_reg (bool): Whether to plot regression results (default: False).
            - plot_confusion (bool): Whether to plot a confusion matrix for classification tasks (default: False).
            - classes (np.array): Data labels if you want to plot confusion at same time
            - Hyperparameters for optimizers

        Returns:
        --------
        dict:
            Dictionary containing training and validation loss for each epoch.
        """
        # Validate input size: Ensure input size matches the first layer's input size (ignoring batch size)
        self.validate_input(X_train)
        self.validate_input(X_val)
        self.validate_final_output(Y_train)
        self.validate_final_output(Y_val)

        loss_train = []
        loss_val = []

        # Initialize optimizer for each layer if has not been defined yet or has changed
        if not hasattr(self, 'Optimizer'):
            self.Optimizer = method
            for layer in self.model:
                layer.optimizer_init(optimizer=method, **kwargs)
        if self.Optimizer != method:
            for layer in self.model:
                layer.optimizer_init(optimizer=method, **kwargs)

        # Placeholder to store the training output
        out_train = None

        # Training loop over the specified number of epochs
        for current_epoch in range(epoch):
            # if out_train is None:
            #     # Perform the forward pass if it's the first epoch
            #     out_train = self(X_train)
            
            # Perform backpropagation
            self.backward(X_train, Y_train, Loss_function,
                          learning_rate=learning_rate, batch_size=batch_size)
            
            # Compute the output again after weight updates
            out_train = self(X_train)

            # Calculate and store the training loss
            loss_train.append(Loss_function.forward(out_train, Y_train, inference=True))
            # Perform validation and calculate validation loss
            out_val = self(X_val)
            loss_val.append(Loss_function.forward(out_val, Y_val, inference=True))

            # Plot the training and validation loss curves
            plot_metrics(epoch, current_epoch+1, loss_train, loss_val,
                 Y_train, out_train, Y_val, out_val, **kwargs)

            # Clear the output to update the plot in real-time
            clear_output(wait=True)

        # Return a dictionary of training and validation losses
        return {'loss_train': loss_train, 'loss_validation': loss_val}

    #################################################################

    def validate_input(self, X: np.ndarray) -> None:
        """
        Validates the input to ensure it matches the expected dimensions and size for the first layer.

        Parameters:
        -----------
        input : np.ndarray
            The input array to validate.

        Raises:
        -------
        ValueError:
            If the input does not have at least 2 dimensions (batch size and input size).
            If the input size does not match the first layer's expected input size.
        """
        # Check if input has at least 2 dimensions (batch size and input size)
        if X.ndim < 2:
            raise ValueError(
                f"Input dimension is too low: Expected at least 2 dimensions (batch size and input size), "
                f"but got {X.ndim} dimensions."
            )

        # Retrieve the first layer's expected input size
        first_layer_input_size = self.model[0].input_size

        # Determine the expected input shape, handling both int and tuple cases
        if isinstance(first_layer_input_size, int):
            expected_input_shape = (first_layer_input_size,)
        else:
            expected_input_shape = first_layer_input_size

        # Validate that the provided input size matches the first layer's expected input size
        if X.shape[1:] != expected_input_shape:
            raise ValueError(
                f"Input size mismatch: Provided input size {X.shape[1:]} "
                f"does not match the first layer's expected input size {expected_input_shape}."
        )

    #################################################################

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Perform a forward pass through the model.

        Parameters:
        -----------
        input : np.ndarray
            The input data to be fed into the model.

        Returns:
        --------
        np.ndarray:
            The output of the model after processing the input.
        """
        # Validate input size: Ensure input size matches the first layer's input size (ignoring batch size)
        self.validate_input(input)

        # Determine how many batches are needed based on batch size
        batch_num = int(np.ceil(input.shape[0] / self.batch_size))

        shape = (input.shape[0], ) + self.model[-1].output_size if self.model[-1].output_size is tuple\
            else (input.shape[0], self.model[-1].output_size)
        out = np.zeros(shape)

        # Process each batch of data
        for i in range(batch_num):
            # Get the batch of input data
            data_X = input[i * self.batch_size: (i + 1) * self.batch_size]
            layer_in = data_X.copy()

            # Forward pass through each layer of the model
            for layer in self.model:
                layer_in = layer(layer_in)
            
            out[i * self.batch_size: (i + 1) * self.batch_size] = layer_in

        # Reshape and return the final output
        return out

    #################################################################

    def Jaccobian(self, X_train: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian matrix for the model given training input data.

        Parameters:
        -----------
        X_train : np.ndarray
            Training data with shape (n_samples, input_size).

        Returns:
        --------
        jaccob : np.ndarray
            The Jacobian matrix with shape (n_samples * output_size, trainable_params).
            Represents partial derivatives of each output with respect to trainable parameters.
        """
        # Validate input size: Ensure input size matches the first layer's input size (ignoring batch size)
        self.validate_input(X_train)

        if type(self.model[-1].output_size) is not int:
            raise TypeError('Jaccobian calculations for the last layer is not supported, use reshaping to see what would happen')
        # Initialize an empty Jacobian matrix with shape (total outputs, total trainable parameters)
        jaccob = np.zeros((X_train.shape[0] * self.model[-1].output_size, self.trainable_params()))

        # Loop over each training sample
        for batch_index in range(X_train.shape[0]):
            # Loop over each output neuron
            for out_ind in range(self.model[-1].output_size):
                # Initialize a vector to isolate one output at a time for differentiation
                E_neuron = np.zeros((1, self.model[-1].output_size))
                E_neuron[:, out_ind] = 1.0
                
                # Temporary variable to store the Jacobian row
                J_row = np.array([])

                # Perform a forward pass with the current input
                _ = self(X_train[batch_index].reshape((1, -1)))
                
                # Backpropagate through each layer in reverse order
                for layer in reversed(self.model):
                    if layer == self.model[0]:  # If the layer is the first one
                        # Get gradients for the first layer without modifying the layer state
                        grads = layer.backward(E_neuron, return_grads=True, modify=False)
                        # Concatenate gradients into the Jacobian row
                        J_row = np.concatenate((np.ravel(grads), J_row))
                    else:  # For all other layers
                        # Perform backpropagation to get errors and gradients
                        back_out = layer.backward(E_neuron, return_error=True, return_grads=True, modify=False)
                        # Update the error for the next layer
                        E_neuron = back_out['error_in']
                        # Get the gradients for the current layer
                        grads = back_out['gradients']
                        # Concatenate gradients into the Jacobian row
                        J_row = np.concatenate((np.ravel(grads), J_row))
                
                # Update the corresponding row in the Jacobian matrix
                ind = batch_index * self.model[-1].output_size + out_ind
                jaccob[ind] = J_row

        return jaccob

    #################################################################

    def levenberg_mar(self, X_train: np.ndarray, X_val: np.ndarray, Y_train: np.ndarray, Y_val: np.ndarray, Loss_function,
                    epoch: int = 15, learning_rate: float = 0.7, gamma: float = 0.99, v: float = 0.9, **kwargs) -> dict:
        """
        Perform the Levenberg-Marquardt optimization algorithm for model training.

        Parameters:
        -----------
        X_train : np.ndarray
            Training input data with shape (n_samples, input_size).
            
        X_val : np.ndarray
            Validation input data with shape (n_samples, input_size).
            
        Y_train : np.ndarray
            Training output data with shape (n_samples, output_size).
            
        Y_val : np.ndarray
            Validation output data with shape (n_samples, output_size).
            
        Loss_function : object
            The loss function object that provides `forward()` and `backward()` methods.
            
        epoch : int, optional
            Number of training epochs. Default is 15.
            
        learning_rate : float, optional
            The learning rate for weight updates. Default is 0.7.
            
        gamma : float, optional
            The damping parameter for the Levenberg-Marquardt algorithm. Default is 0.8.
            
        v : float, optional
            The adjustment factor for `gamma`. Default is 0.9.
            
        **kwargs : dict
            Additional keyword arguments for plotting or other purposes.

        Returns:
        --------
        dict
            A dictionary containing training and validation loss history:
            {'loss_train': list, 'loss_validation': list}
        """
        # Validate input size: Ensure input size matches the first layer's input size (ignoring batch size)
        self.validate_input(X_train)
        self.validate_input(X_val)
        self.validate_final_output(Y_train)
        self.validate_final_output(Y_val)

        if type(self.model[-1].output_size) is not int:
            raise TypeError('Jaccobian calculations for the last layer is not supported, use reshaping to see what would happen')
        
        # Initialize each layer's optimizer to SGD to avoid momentum effects (e.g., Adam)
        for layer in self.model:
            layer.optimizer_init('SGD')

        # Lists to store training and validation losses
        loss_train = []
        loss_val = []

        # Training loop for the specified number of epochs
        for current_epoch in range(epoch):
            # Adjust the damping parameter `gamma` based on recent training loss
            if current_epoch >= 2:
                if loss_train[-1] >= loss_train[-2]:
                    gamma /= v
                else:
                    gamma *= v

            # Forward pass through the model with training data
            out = self(X_train)
            
            # Reshape training data to match input size
            # X_train = X_train.reshape(-1, self.model[0].input_size)
            # Y_train = Y_train.reshape(out.shape)

            # Compute the error using the provided loss function
            if hasattr(Loss_function, 'memory'):
                error = np.zeros(Y_train.shape)
                for ind, o in enumerate(out):
                    _ = Loss_function.forward(o.reshape((1,-1)), Y_train[ind].reshape((1,-1)))
                    error[ind] = Loss_function.backward()
            else:
                error = Loss_function.backward(out, Y_train)

            # Compute the Jacobian matrix for the current training data
            J = self.Jaccobian(X_train)

            # Compute the gradient update using the Levenberg-Marquardt formula
            new_grads = np.linalg.inv((J.T @ J + gamma * np.eye(self.trainable_params()))) @ J.T @ error.reshape((-1, 1))

            # Update model weights using the computed gradients
            ind2 = 0
            for layer in self.model:
                ind1 = ind2
                ind2 += layer.trainable_params()
                layer.update(new_grads[ind1:ind2].reshape((-1, 1)), learning_rate)

            # Forward pass for training data to compute training loss
            out_train = self(X_train)
            loss_train.append(Loss_function.forward(out_train, Y_train, inference=True))

            # Forward pass for validation data to compute validation loss
            out_val = self(X_val)
            loss_val.append(Loss_function.forward(out_val, Y_val, inference=True))

            # Plot training and validation metrics for visual feedback
            plot_metrics(epoch, current_epoch + 1, loss_train, loss_val, Y_train, out_train, Y_val, out_val, **kwargs)

            # Clear output to refresh the plot in real-time
            clear_output(wait=True)

        # Return the training and validation loss history as a dictionary
        return {'loss_train': loss_train, 'loss_validation': loss_val}

    #################################################################

    def EKF(self, X_train: np.ndarray, X_val: np.ndarray, Y_train: np.ndarray, Y_val: np.ndarray,
            Loss_function, epoch: int = 15, Q: np.ndarray = None, R: np.ndarray = None, P: np.ndarray = None, 
            learning_rate: float = 1.0, **kwargs) -> dict:
        """
        Perform Extended Kalman Filter (EKF) optimization for training the model.

        Parameters:
        -----------
        X_train : np.ndarray
            Training input data of shape (n_samples, input_size).
        X_val : np.ndarray
            Validation input data of shape (n_samples, input_size).
        Y_train : np.ndarray
            Training target data of shape (n_samples, output_size).
        Y_val : np.ndarray
            Validation target data of shape (n_samples, output_size).
        Loss_function : object
            The loss function object with `forward()` and `backward()` methods.
        epoch : int, optional
            Number of training epochs. Default is 15.
        Q : np.ndarray, optional
            Process noise covariance matrix. If None, a default diagonal matrix is used.
        R : np.ndarray, optional
            Measurement noise covariance matrix. If None, a default diagonal matrix is used.
        P : np.ndarray, optional
            Error covariance matrix. If None, a default diagonal matrix is used.
        learning_rate : float, optional
            The learning rate for parameter updates. Default is 1.0.
        **kwargs : dict
            Additional keyword arguments for visualization or other functionalities.

        Returns:
        --------
        dict
            Dictionary containing training and validation loss history:
            {'loss_train': list, 'loss_validation': list}.
        """
        # Validate input size: Ensure input size matches the first layer's input size (ignoring batch size)
        self.validate_input(X_train)
        self.validate_input(X_val)
        self.validate_final_output(Y_train)
        self.validate_final_output(Y_val)

        # Initialize covariance matrices
        Q = Q * np.eye(self.trainable_params()) if Q is not None else 0.1 * np.eye(self.trainable_params())
        R = R * np.eye(self.model[-1].output_size) if R is not None else 0.001 * np.eye(self.model[-1].output_size)
        P = P * np.eye(self.trainable_params()) if P is not None else 0.01 * np.eye(self.trainable_params())

        # Initialize lists to store training and validation losses
        loss_train = []
        loss_val = []

        # Ensure the optimizer for each layer is set to SGD
        for layer in self.model:
            layer.optimizer_init('SGD')

        # Get the number of training samples
        n_samples = X_train.shape[0]

        # Loop through each training epoch
        for current_epoch in range(epoch):
            # Iterate through each sample in the training data
            for i in range(n_samples):
                # Extract the current input and target sample
                data_X = X_train[i:i + 1].copy()  # Shape preservation for single sample
                data_Y = Y_train[i:i + 1].copy()

                # Perform a forward pass through the model
                out = self(data_X)

                # Compute the error using the loss function
                if hasattr(Loss_function, 'memory'):
                    _ = Loss_function.forward(out, data_Y)
                    error = Loss_function.backward()
                else:
                    error = Loss_function.backward(out, data_Y)

                # Calculate the Jacobian matrix for the current input
                H = self.Jaccobian(data_X).T

                # Compute the normalization matrix (R + H^T * P * H)
                A = np.linalg.inv(R + H.T @ P @ H)

                # Compute the Kalman gain (K)
                K = P @ H @ A

                # Calculate gradient updates (new_grads)
                new_grads = K @ error.T

                # Update the error covariance matrix (P)
                P -= K @ H.T @ P - Q

                # Update model parameters using the calculated gradients
                ind2 = 0
                for layer in self.model:
                    ind1 = ind2
                    ind2 += layer.trainable_params()
                    layer.update(new_grads[ind1:ind2].reshape((-1, 1)), learning_rate)

            # Compute the training loss for the current epoch
            out_train = self(X_train)
            loss_train.append(Loss_function.forward(out_train, Y_train, inference=True))

            # Compute the validation loss for the current epoch
            out_val = self(X_val)
            loss_val.append(Loss_function.forward(out_val, Y_val, inference=True))

            # Plot training and validation metrics
            plot_metrics(epoch, current_epoch + 1, loss_train, loss_val, Y_train, out_train, Y_val, out_val, **kwargs)

            # Clear the output to update the plot dynamically
            clear_output(wait=True)

        # Return a dictionary with training and validation loss histories
        return {'loss_train': loss_train, 'loss_validation': loss_val}