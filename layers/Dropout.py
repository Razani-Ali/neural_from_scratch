import numpy as np
from activations.activation_functions import net2out, net2Fprime
from initializers.weight_initializer import Dense_weight_init
from optimizers.set_optimizer import init_optimizer


class Dropout:
    """
    Dropout layer implementation for neural networks.

    Parameters:
    rate (float): The probability of dropping a unit during training. Default is 0.2.
    """
    def __init__(self, rate: float = 0.2) -> None:
        """
        Initializes the Dropout layer with the specified drop rate.

        Attributes:
        rate (float): The dropout rate.
        activation (str): A string representation of the dropout activation.
        """
        self.rate = rate  # Set the dropout rate
        self.activation = f"None, drop rate is {rate}"  # Activation description

    #################################################################

    def trainable_params(self) -> int:
        """
        Returns the number of trainable parameters in the Dropout layer.

        Returns:
        int: Number of trainable parameters (always 0 for Dropout).
        """
        return 0

    #################################################################

    def all_params(self) -> int:
        """
        Returns the total number of parameters in the Dropout layer.

        Returns:
        int: Total number of parameters (always 0 for Dropout).
        """
        return 0

    #################################################################

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Applies dropout to the input during training.

        Parameters:
        input (np.ndarray): Input array of shape (batch size, input size).

        Returns:
        np.ndarray: Output array after applying the dropout mask.

        Raises:
        ValueError: If input is not a 2D array.
        """
        if np.ndim(input) != 2:
            raise ValueError("Dropout only supports an input of shape (batch size, input size)")  # Ensure input is 2D
        
        self.mask = np.random.uniform(size=input.shape) > self.rate  # Generate dropout mask
        return self.mask * input  # Apply mask to input

    #################################################################

    def optimizer_init(self, optimizer: str = 'Adam', **kwargs) -> None:
        """
        Placeholder for initializing an optimizer. No implementation needed for Dropout.

        Parameters:
        optimizer (str): Optimizer name (default: 'Adam').
        kwargs: Additional optimizer parameters.
        """
        pass

    #################################################################

    def update(self, grads: np.ndarray, learning_rate: float = 1e-3) -> None:
        """
        Placeholder for updating layer parameters. No implementation needed for Dropout.

        Parameters:
        grads (np.ndarray): Gradients (not used in Dropout).
        learning_rate (float): Learning rate (default: 1e-3, not used in Dropout).
        """
        pass

    #################################################################

    def backward(self, 
                 error_batch: np.ndarray, 
                 return_error: bool = False, 
                 return_grads: bool = False, 
                 **kwargs) -> dict | np.ndarray | list | None:
        """
        Backward pass through the Dropout layer.

        Parameters:
        error_batch (np.ndarray): The error from the subsequent layer.
        return_error (bool): Whether to return the propagated error (default: False).
        return_grads (bool): Whether to return gradients (default: False).
        kwargs: Additional parameters (not used).

        Returns:
        dict | np.ndarray | list | None:
            - If return_error and return_grads are True, returns a dict with error and empty gradients.
            - If return_error is True, returns the propagated error.
            - If return_grads is True, returns an empty list.
            - If both are False, returns None.
        """
        if return_error:
            error_in = self.mask * error_batch  # Apply dropout mask to error

        if return_error and return_grads:
            return {'error_in': error_in, 'gradients': []}  # Return error and empty gradients
        elif return_error:
            return error_in  # Return propagated error only
        elif return_grads:
            return []  # Return empty gradients
