import numpy as np


class Reshape:
    """
    A Reshape layer for neural networks.

    This layer reshapes the input array into a specified output shape. It does not
    contain any trainable parameters, as it only performs a reshaping operation.

    Attributes:
    ----------
    input_size : tuple[int, ...] | int
        Shape of the input data.
    output_size : tuple[int, ...] | int
        Shape of the output data.
    batch_size : int
        Number of samples in each batch.
    activation : str
        The activation function used (fixed to 'None' for this layer).
    """

    def __init__(self, input_shape: tuple[int, ...] | int, output_shape: tuple[int, ...] | int, batch_size: int = 32) -> None:
        """
        Initialize the Reshape layer.

        Parameters:
        ----------
        input_shape : tuple[int, ...] | int
            The shape of the input data.
        output_shape : tuple[int, ...] | int
            The desired shape of the output data.
        batch_size : int
            Number of samples in each batch (default is 32).
        """
        self.input_size = input_shape  # Store the input shape
        self.output_size = output_shape  # Store the output shape
        self.batch_size = batch_size  # Store batch size
        self.activation = 'None'  # No activation function for this layer

    #################################################################

    def trainable_params(self) -> int:
        """
        Returns the number of trainable parameters in the layer.

        Since this layer only reshapes the data, it has no trainable parameters.

        Returns:
        -------
        int: Total number of trainable parameters (always 0).
        """
        return 0  # No trainable parameters exist

    #################################################################

    def all_params(self) -> int:
        """
        Returns the total number of parameters in the layer.

        Since this layer only reshapes the data, it has no parameters at all.

        Returns:
        -------
        int: Total number of parameters (always 0).
        """
        return 0  # No parameters (trainable or otherwise) exist

    #################################################################

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the Reshape layer.

        Parameters:
        ----------
        input : np.ndarray
            Input data array of shape `self.input_size`.

        Returns:
        -------
        np.ndarray: Output data array reshaped to `self.output_size`.
        """
        # Reshape the input to the desired output size and return
        shape = (-1,) + self.output_size if isinstance(self.output_size, tuple) else (-1, self.output_size)
        return input.reshape(shape)

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        A placeholder for optimizer initialization.

        Since the layer has no trainable parameters, this method does nothing.

        Parameters:
        ----------
        optimizer : str, optional
            Name of the optimizer (default is 'Adam').
        kwargs : dict, optional
            Additional arguments for the optimizer.
        """
        pass  # No initialization required for reshaping

    #################################################################

    def update(self, grads: np.ndarray, learning_rate: float = 1e-3) -> None:
        """
        A placeholder for updating trainable parameters.

        Since the layer has no trainable parameters, this method does nothing.

        Parameters:
        ----------
        grads : np.ndarray
            Gradients to be applied (not used).
        learning_rate : float, optional
            Learning rate for updates (default is 1e-3).
        """
        pass  # No parameters to update

    #################################################################

    def backward(
        self,
        error_batch: np.ndarray,
        return_error: bool = False,
        return_grads: bool = False,
        **kwargs
    ) -> dict | np.ndarray | list:
        """
        Performs a backward pass through the Reshape layer.

        The backward pass reshapes the error to match the input size of the layer.

        Parameters:
        ----------
        error_batch : np.ndarray
            Error signal from the next layer, of shape `self.output_size`.
        return_error : bool, optional
            Whether to return the reshaped error signal (default is False).
        return_grads : bool, optional
            Whether to return the gradients (default is False, always empty list).

        Returns:
        -------
        dict | np.ndarray | list:
            - If both `return_error` and `return_grads` are True:
              Returns a dictionary with `error_in` (reshaped error signal) and `gradients` (empty list).
            - If only `return_error` is True:
              Returns the reshaped error signal as `np.ndarray`.
            - If only `return_grads` is True:
              Returns an empty list.
            - If neither is True:
              Returns `None`.
        """
        # Reshape the error signal to match the input size of the layer
        shape = (-1,) + self.input_size if isinstance(self.input_size, tuple) else (-1, self.input_size)
        reshaped_error = error_batch.reshape(shape)

        # Return both error and gradients if requested
        if return_error and return_grads:
            return {'error_in': reshaped_error, 'gradients': []}

        # Return only the error signal if requested
        elif return_error:
            return reshaped_error

        # Return only the gradients (always empty) if requested
        elif return_grads:
            return []

        # Return nothing if neither flag is set
        return None