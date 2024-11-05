import numpy as np


class RMSprop:
    """
    Custom implementation of the RMSprop optimizer for gradient-based optimization.
    Stores the state of the moving average of the squared gradients for each parameter
    and updates them based on the gradients provided.

    Attributes:
    -----------
    num_params : int
        Number of parameters to be optimized.
    beta : float
        Exponential decay rate for the moving average of squared gradients.
    eps : float
        Small constant to prevent division by zero in the update rule.
    vt : np.ndarray
        Array for the moving average of squared gradients, initialized to zeros.

    Methods:
    --------
    __call__(grads: np.ndarray, learning_rate: float) -> np.ndarray
        Calculates parameter updates based on current gradients and learning rate.

    reset()
        Resets the moving average of squared gradients to initial values.
    """

    def __init__(self, num_params: int, beta: float, eps: float):
        """
        Initializes RMSprop optimizer parameters, including the moving average
        of squared gradients and decay rate.

        Parameters:
        -----------
        num_params : int
            The number of parameters in the model that will be optimized.
        beta : float
            The decay rate for the moving average of squared gradients.
        eps : float
            A small constant added to avoid division by zero.
        """
        self.num_params = num_params  # Store the number of parameters to optimize

        # Initialize the moving average of squared gradients if number of parameters > 0
        if num_params != 0:
            self.vt = np.zeros((num_params, 1))  # Moving average of squared gradients
            self.beta = beta  # Decay rate for the moving average
            self.eps = eps  # Small constant for numerical stability

    def __call__(self, grads: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Updates the moving average of squared gradients and calculates parameter update values.

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
            # Update the moving average of squared gradients (vt) with exponential decay
            self.vt = self.beta * self.vt + (1 - self.beta) * np.square(grads)

            # Compute parameter updates (delta) using the root of the average squared gradient
            # and the learning rate
            delta = learning_rate * grads / (np.sqrt(self.vt) + self.eps)
            return delta  # Return computed updates for parameters

    def reset(self):
        """
        Resets the optimizer's moving average of squared gradients to initial values,
        clearing previous optimization states.
        """
        # Only reset if there are parameters to reset
        if self.num_params != 0:
            self.vt = np.zeros(self.vt.shape)  # Reset moving average of squared gradients to zero