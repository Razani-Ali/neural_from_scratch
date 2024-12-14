import numpy as np

class Jordan:
    """
    Jordan class implements a recurrent neural network (RNN) architecture that integrates sequential data processing 
    with feedback mechanisms, specifically a Jordan-type feedback structure. This class allows for stateful training, 
    returning intermediate states, and supports model optimization and backpropagation.

    Parameters:
    -----------
    model : list
        A list of neural network layers that form the model.
    batch_size : int, optional
        The size of data batches for training and inference (default is 32).
    stateful : bool, optional
        Whether to maintain the state between batches (default is False).
    return_states : bool, optional
        If True, returns all intermediate states; otherwise, only the final output is returned (default is False).
    """

    def __init__(self, model, batch_size: int = 32, stateful: bool = False, return_states: bool = False):
        self.model = model  # List of model layers
        self.output_size = (model[0].time_steps, model[-1].output_size) if return_states else (1, model[-1].output_size)
        self.input_size = (model[0].time_steps, model[0].input_size)
        self.time_steps = model[0].time_steps  # Total time steps for sequences
        self.batch_size = batch_size  # Batch size for processing
        self.activation = 'None'  # Default activation function
        self.pervious_batch_state = np.zeros((model[-1].output_size, 1)) if stateful else None
        self.batch_states = np.zeros((batch_size, model[0].time_steps, model[-1].output_size, 1)) if stateful else \
            np.zeros((model[0].time_steps, model[-1].output_size, 1))
        self.stateful = stateful  # Whether to maintain state between batches
        self.return_states = return_states  # Whether to return all states or only the final output

    #################################################################

    def trainable_params(self) -> int:
        """
        Compute the total number of trainable parameters across all layers in the model.

        Returns:
        --------
        int
            The total count of trainable parameters.
        """
        params = 0
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
        params = 0
        for layer in self.model:
            params += layer.all_params()
        return int(params)

    #################################################################

    def summary(self) -> None:
        """
        Print a summary of the model, including details about each layer, their parameters, 
        and the total parameters in the model.
        """
        print('\n', '*' * 30, 'model summary', '*' * 30, '\n')
        total_n_trainable = 0
        total_n_all = 0
        for index, layer in enumerate(self.model):
            print(f'layer {index+1}:', end='\n\t')
            print(type(layer), end='\n\t')
            print('activation function:', layer.activation, end='\n\t')
            print('batch size:', layer.batch_size, end='\n\t')
            print('sequence length:', layer.time_steps, end='\n\t')
            print('input size:', layer.input_size, end='\n\t')
            print('output size:', layer.output_size, end='\n\t')
            n_trainable = layer.trainable_params()
            total_n_trainable += n_trainable
            n_all = layer.all_params()
            total_n_all += n_all
            print(f'number of parameters: {n_all}', end='\n\t')
            print(f'number of trainable parameters: {n_trainable}', end='\n\t')
            print(f'number of non trainable parameters: {n_all - n_trainable}', end='\n\t')
            print('-' * 50)
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
            The final output or states of the model depending on return_states.
        """
        input = input.reshape((-1, self.time_steps, self.input_size[1], 1))
        self.input_shape = input.shape  # Save input shape for backward pass
        input_batch_size = input.shape[0]
        if self.batch_size < input_batch_size:
            raise ValueError('Data batch size cannot be larger than model batch size')

        state = self.pervious_batch_state if self.stateful else np.zeros((self.model[-1].output_size, 1))
        for batch_index, sequence in enumerate(input):
            for seq_index, time_step in enumerate(sequence):
                X = time_step.copy()
                for layer_index, layer in enumerate(self.model):
                    if layer_index == 0:
                        # First layer processes input and state
                        X = layer(batch_index, seq_index, X, state)
                    elif layer_index == len(self.model) - 1:
                        # Last layer processes input and updates state
                        X = layer(batch_index, seq_index, X)
                        state = X.copy()
                        if self.stateful:
                            self.batch_states[batch_index, seq_index] = state
                        else:
                            self.batch_states[seq_index] = state
                    else:
                        # Intermediate layers process the output of the previous layer
                        X = layer(batch_index, seq_index, X)
            state = state if self.stateful else np.zeros(state.shape)
        self.pervious_batch_state = state

        if self.return_states:
            return self.model[-1].output[:input_batch_size, :, :, 0].reshape((-1, self.time_steps, self.model[-1].output_size, 1))
        else:
            return self.model[-1].output[:input_batch_size, -1].reshape((-1, self.model[-1].output_size, 1))

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
        if not hasattr(self, 'Optimizer'):
            self.Optimizer = optimizer
            for layer in self.model:
                layer.optimizer_init(optimizer=optimizer, **kwargs)
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
        ind2 = 0
        for layer in self.model:
            ind1 = ind2
            ind2 += layer.trainable_params()
            layer.update(batch_size=1, grads=grads[ind1:ind2].reshape((-1, 1)), learning_rate=learning_rate)

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
        if self.return_states:
            error_batch = error_batch.reshape((-1, self.time_steps, self.model[-1].output_size, 1))
        else:
            error_batch = error_batch.reshape((self.input_shape[0], 1, self.model[-1].output_size, 1))
            zeros = np.zeros((self.input_shape[0], self.time_steps-1, self.model[-1].output_size, 1))
            error_batch = np.concatenate((error_batch, zeros), axis=1)

        if return_error:
            error_in = np.zeros(self.input_shape)

        batch_size, seq_size = len(error_batch), self.model[0].time_steps
        time_E = np.zeros((self.model[-1].output_size, 1))

        for batch_index in reversed(range(batch_size)):
            time_E = time_E if self.stateful else time_E * 0
            for seq_index in reversed(range(seq_size)):
                E = error_batch[batch_index, seq_index]
                for layer_index in reversed(range(len(self.model))):
                    layer = self.model[layer_index]
                    if layer_index == 0:
                        # Compute gradients and error for the first layer
                        state = self.batch_states[batch_index, seq_index] if self.stateful else self.batch_states[seq_index]
                        E, time_E = layer.backward(batch_index, seq_index, E, state)
                    elif layer_index == len(self.model) - 1:
                        # Compute gradients and error for the last layer
                        E += time_E
                        E = layer.backward(batch_index, seq_index, E)
                    else:
                        # Compute gradients and error for intermediate layers
                        E = layer.backward(batch_index, seq_index, E)
                if return_error:
                    error_in[batch_index, seq_index] = E.reshape((-1, 1))

        if return_grads:
            grads = None if self.trainable_params() == 0 else np.array([]).reshape((-1, 1))
            for layer in self.model:
                grad = layer.return_grads()
                if grad is not None:
                    grads = np.concatenate((grads, grad.reshape((-1, 1))))

        if modify:
            for layer in self.model:
                layer.update(self.input_shape[0], learning_rate=learning_rate)

        if return_error and return_grads:
            return {'error_in': error_in, 'gradients': grads}
        elif return_error:
            return error_in
        elif return_grads:
            return grads