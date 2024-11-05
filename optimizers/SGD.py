import numpy as np


class SGD:
    """
    Custom implementation of the SGD (Stochastic Gradient Descent) optimizer
    for gradient-based optimization. This optimizer updates parameters based
    directly on the gradients without any moment accumulation.

    Attributes:
    -----------
    num_params : int
        Number of parameters to be optimized.

    Methods:
    --------
    __call__(grads: np.ndarray, learning_rate: float) -> np.ndarray
        Calculates parameter updates based on current gradients and learning rate.

    reset()
        Placeholder method to maintain compatibility with other optimizers.
    """

    def __init__(self, num_params: int):
        """
        Initializes the SGD optimizer parameters.

        Parameters:
        -----------
        num_params : int
            The number of parameters in the model that will be optimized.
        """
        self.num_params = num_params  # Store the number of parameters to optimize

    def __call__(self, grads: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Calculates parameter updates directly based on the gradients and learning rate.

        Parameters:
        -----------
        grads : np.ndarray
            The gradient values for each parameter (size matches num_params).
        learning_rate : float
            The learning rate controlling the step size of updates.

        Returns:
        --------
        delta : np.ndarray
            Array of parameter update values.
        """
        if grads is not None:
            # Compute parameter updates (delta) by directly scaling the gradients with the learning rate
            delta = learning_rate * grads
            return delta  # Return computed updates for parameters

    def reset(self):
        """
        Placeholder method to maintain compatibility with other optimizers.
        Since SGD does not maintain any state, no reset is required.
        """
        pass  # No state to reset for SGD