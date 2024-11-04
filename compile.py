import numpy as np
# from layers.Dense_network import Dense
# from layers.Rough_network import Rough
from layers.RBF_network import RBF
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
        # Reshape input to ensure compatibility with the model's input size
        input = input.reshape(-1, self.model[0].input_size)
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
            error = Loss_function.backward(out, data_Y).reshape(-1, self.model[-1].output_size)

            # Perform backpropagation on each layer, starting from the last layer
            for layer in reversed(self.model):
                if layer == self.model[0]:
                    layer.backward(error, learning_rate=learning_rate, return_error=True)
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
        loss_train = []
        loss_val = []

        # Initialize optimizer for each layer if has not been defined yet or has changed
        if not hasattr(self, 'Optimizer'):
            self.Optimizer = method
            for layer in self.model:
                layer.optimzer_init(optimizer=method, **kwargs)
        if self.Optimizer != method:
            for layer in self.model:
                layer.optimzer_init(optimizer=method, **kwargs)

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
            loss_train.append(Loss_function.forward(out_train, Y_train))

            # Perform validation and calculate validation loss
            out_val = self(X_val)
            loss_val.append(Loss_function.forward(out_val, Y_val))

            # Plot the training and validation loss curves
            plot_metrics(epoch, current_epoch+1, loss_train, loss_val,
                 Y_train, out_train, Y_val, out_val, **kwargs)

            # Clear the output to update the plot in real-time
            clear_output(wait=True)

        # Return a dictionary of training and validation losses
        return {'loss_train': loss_train, 'loss_validation': loss_val}

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
        # Ensure input is reshaped to match the model's expected input size
        input = input.reshape((-1, self.model[0].input_size))

        # Determine how many batches are needed based on batch size
        batch_num = int(np.ceil(input.shape[0] / self.model[0].batch_size))

        out = np.zeros((input.shape[0], self.model[-1].output_size))

        # Process each batch of data
        for i in range(batch_num):
            # Get the batch of input data
            data_X = input[i * self.model[0].batch_size: (i + 1) * self.model[0].batch_size]
            layer_in = data_X.copy()

            # Forward pass through each layer of the model
            for layer in self.model:
                layer_in = layer(layer_in)
            
            out[i * self.model[0].batch_size: (i + 1) * self.model[0].batch_size] = layer_in

        # Reshape and return the final output
        return out
