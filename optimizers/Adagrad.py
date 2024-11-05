import numpy as np


class Adagrad:
    """
    Custom implementation of the Adagrad optimizer for gradient-based optimization.
    Stores a cumulative sum of squared gradients for each parameter, allowing the
    learning rate to adjust for each parameter individually based on past updates.

    Attributes:
    -----------
    num_params : int
        Number of parameters to be optimized.
    eps : float
        Small constant to prevent division by zero in the update rule.
    gt : np.ndarray
        Array for the cumulative sum of squared gradients, initialized to zeros.

    Methods:
    --------
    __call__(grads: np.ndarray, learning_rate: float) -> np.ndarray
        Calculates parameter updates based on current gradients and learning rate.

    reset()
        Resets the cumulative sum of squared gradients to initial values.
    """

    def __init__(self, num_params: int, eps: float):
        """
        Initializes Adagrad optimizer parameters, including the cumulative squared
        gradient sum and a small constant for numerical stability.

        Parameters:
        -----------
        num_params : int
            The number of parameters in the model that will be optimized.
        eps : float
            A small constant added to avoid division by zero.
        """
        self.num_params = num_params  # Store the number of parameters to optimize

        # Initialize the cumulative sum of squared gradients if number of parameters > 0
        if num_params != 0:
            self.gt = np.zeros((num_params, 1))  # Cumulative sum of squared gradients
            self.eps = eps  # Small constant for numerical stability

    def __call__(self, grads: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Updates the cumulative sum of squared gradients and calculates parameter update values.

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
            # Update the cumulative sum of squared gradients (gt)
            self.gt += np.square(grads)

            # Compute parameter updates (delta) using the adaptive learning rate for each parameter
            delta = learning_rate * grads / (np.sqrt(self.gt) + self.eps)
            return delta  # Return computed updates for parameters

    def reset(self):
        """
        Resets the optimizer's cumulative sum of squared gradients to initial values,
        clearing previous optimization states.
        """
        # Only reset if there are parameters to reset
        if self.num_params != 0:
            self.gt = np.zeros(self.gt.shape)  # Reset cumulative squared gradient sum to zero