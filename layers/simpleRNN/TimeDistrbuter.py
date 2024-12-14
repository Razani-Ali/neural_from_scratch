import numpy as np

class TimeDistributer:
    """
    TimeDistributer class allows sequential processing of input data across a given neural network model. 
    It distributes the time steps of the input data to the model layers and handles the forward and backward passes.

    Parameters:
    -----------
    model : list
        A list of neural network layers that form the model.
    batch_size : int, optional
        The size of data batches for training and inference (default is 32).
    """

    def __init__(self, model, batch_size: int = 32):
        # Validate layer compatibility: Check if each layer's output size matches the next layer's input size
        for i in range(len(model) - 1):
            if model[i].output_size != model[i + 1].input_size:
                raise ValueError(
                    f"Layer mismatch detected: Layer {i + 1} output size ({model[i].output_size}) "
                    f"does not match Layer {i + 2} input size ({model[i + 1].input_size})."
                )
            
        # Retrieve the time_steps of the first layer as the reference
        reference_time_steps = model[0].time_steps

        # Validate time_steps attribute across all layers
        for index, layer in enumerate(model):
            if not hasattr(layer, 'time_steps'):
                raise AttributeError(
                    f"Layer validation failed at Layer {index+1}: Ensure all the layer are 'Time...' layers with a 'time_steps' attribute."
                )
            if layer.time_steps != reference_time_steps:
                raise ValueError(
                    f"Inconsistent time_steps: Layer {index + 1} has time_steps={layer.time_steps}, "
                    f"but expected time_steps={reference_time_steps}. Ensure all layers share the same time_steps."
                )
        
        # Store the model layers
        self.model = model

        # Output size derived from the model
        self.output_size = (model[0].time_steps, model[-1].output_size)

        # Input size derived from the model
        self.input_size = (model[0].time_steps, model[0].input_size)

        # Number of time steps for sequences
        self.time_steps = model[0].time_steps

        # Batch size for processing
        self.batch_size = batch_size

        # Activation function for the model (default: None)
        self.activation = 'None'

    #################################################################

    def trainable_params(self) -> int:
        """
        Compute the total number of trainable parameters across all layers in the model.

        Returns:
        --------
        int
            The total count of trainable parameters.
        """
        # Initialize parameter count
        params = 0

        # Accumulate trainable parameters from all layers
        for layer in self.model:
            params += layer.trainable_params()

        return int(params)

    #################################################################

    def all_params(self) -> int:
        """
        Compute the total number of parameters (trainable and non-trainable) in the model.

        Returns:
        --------
        int
            The total count of all parameters.
        """
        # Initialize parameter count
        params = 0

        # Accumulate all parameters from all layers
        for layer in self.model:
            params += layer.all_params()

        return int(params)

    #################################################################

    def summary(self) -> None:
        """
        Print a summary of the model, including details about each layer, their parameters, 
        and the total parameters in the model.
        """
        # Header for the summary
        print('\n', '*' * 30, 'model summary', '*' * 30, '\n')

        # Initialize total counts
        total_n_trainable = 0
        total_n_all = 0

        # Loop through each layer and print details
        for index, layer in enumerate(self.model):
            print(f'layer {index + 1}:', end='\n\t')
            print(type(layer), end='\n\t')
            print('activation function:', layer.activation, end='\n\t')
            print('batch size:', layer.batch_size, end='\n\t')
            print('sequence length:', layer.time_steps, end='\n\t')
            print('input size:', layer.input_size, end='\n\t')
            print('output size:', layer.output_size, end='\n\t')

            # Count trainable and all parameters
            n_trainable = layer.trainable_params()
            total_n_trainable += n_trainable
            n_all = layer.all_params()
            total_n_all += n_all

            # Print parameter counts
            print(f'number of parameters: {n_all}', end='\n\t')
            print(f'number of trainable parameters: {n_trainable}', end='\n\t')
            print(f'number of non trainable parameters: {n_all - n_trainable}', end='\n\t')
            print('-' * 50)

        # Print totals
        print(f'total number of parameters: {total_n_all}', end='\n\t')
        print(f'total number of trainable parameters: {total_n_trainable}', end='\n\t')
        print(f'total number of non trainable parameters: {total_n_all - total_n_trainable}', end='\n\t')
        
    #################################################################

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass of the model on the given input data.

        Parameters:
        -----------
        input : np.ndarray
            The input data, shaped as (batch_size, time_steps, input_features).

        Returns:
        --------
        np.ndarray
            The final output of the model, reshaped to include time steps.
        """
        # Save the input shape for backward pass
        self.input_shape = input.shape

        # Get the batch size from the input
        input_batch_size = input.shape[0]

        # Check if the input batch size exceeds the model's batch size
        if self.batch_size < input_batch_size:
            raise ValueError('Data batch size cannot be larger than model batch size')

        # Iterate through batches
        for batch_index, sequence in enumerate(input):
            # Iterate through time steps in the sequence
            for seq_index, time_step in enumerate(sequence):
                # Copy the time step input
                X = time_step.reshape((-1,1))

                # Pass the input through each layer in the model
                for layer in self.model:
                    X = layer(batch_index, seq_index, X)

        # Return the final output reshaped for time steps
        return self.model[-1].output[:input_batch_size, :, :, 0].reshape((-1, self.time_steps, self.model[-1].output_size, 1))

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Initialize the optimizer for all layers in the model.

        Parameters:
        -----------
        optimizer : str, optional
            The type of optimizer to use (default is 'Adam').
        **kwargs : dict, optional
            Additional parameters for the optimizer.

        Returns:
        --------
        None
        """
        # If the optimizer is not already initialized, set it up
        if not hasattr(self, 'Optimizer'):
            self.Optimizer = optimizer
            for layer in self.model:
                layer.optimizer_init(optimizer=optimizer, **kwargs)

        # If a different optimizer is provided, reinitialize for all layers
        if self.Optimizer != optimizer:
            for layer in self.model:
                layer.optimizer_init(optimizer=optimizer, **kwargs)

    #################################################################

    def update(self, grads: np.ndarray, learning_rate: float = 1e-3) -> None:
        """
        Update the parameters of all layers using the provided gradients.

        Parameters:
        -----------
        grads : np.ndarray
            The gradients for all trainable parameters in the model.
        learning_rate : float, optional
            The learning rate for parameter updates (default is 1e-3).

        Returns:
        --------
        None
        """
        # Initialize index for parameter tracking
        ind2 = 0

        # Iterate through each layer and update its parameters
        for layer in self.model:
            ind1 = ind2
            ind2 += layer.trainable_params()
            layer.update(grads[ind1:ind2].reshape((-1, 1)), learning_rate)

    #################################################################

    def backward(self, error_batch: np.ndarray, learning_rate: float = 1e-3,
                 return_error: bool = False, return_grads: bool = False, modify: bool = True):
        """
        Perform backpropagation through the model to compute gradients and optionally update parameters.

        Parameters:
        -----------
        error_batch : np.ndarray
            The error signal to backpropagate.
        learning_rate : float, optional
            The learning rate for parameter updates (default is 1e-3).
        return_error : bool, optional
            If True, return the propagated error signal (default is False).
        return_grads : bool, optional
            If True, return the computed gradients (default is False).
        modify : bool, optional
            If True, update the parameters of the model layers (default is True).

        Returns:
        --------
        dict, np.ndarray, or None
            Depending on the flags, returns propagated errors and/or gradients.
        """
        # If return_error is requested, initialize the error storage
        if return_error:
            error_in = np.zeros(self.input_shape)

        # Iterate through each batch
        for batch_index, sequence_error in enumerate(error_batch):
            # Iterate through each time step in reverse
            for seq_index, time_step_error in enumerate(sequence_error):
                # Copy the error at the current time step
                E = time_step_error.reshape((-1,1))

                # Backpropagate through the layers in reverse order
                for layer in reversed(self.model):
                    E = layer.backward(batch_index, seq_index, E).reshape((-1,1))

                # Store the propagated error if requested
                if return_error:
                    error_in[batch_index, seq_index] = E.ravel()

        # If gradients are requested, accumulate them from all layers
        if return_grads:
            grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
            for layer in self.model:
                grad = layer.return_grads()
                if grad is not None:
                    grads = np.concatenate((grads, grad.reshape((-1, 1))))

        # Update the parameters of all layers if modify is True
        if modify:
            for layer in self.model:
                layer.update(self.input_shape[0], learning_rate=learning_rate)

        # Return the requested outputs
        if return_error and return_grads:
            return {'error_in': error_in, 'gradients': grads}
        elif return_error:
            return error_in
        elif return_grads:
            return grads