import numpy as np


class Momentum:
    """
    Custom implementation of the SGD optimizer with momentum for gradient-based optimization.
    Stores a velocity term for each parameter that accumulates past gradients, allowing
    the optimizer to "build up speed" in areas of consistent gradient direction.

    Attributes:
    -----------
    num_params : int
        Number of parameters to be optimized.
    momentum : float
        Momentum factor controlling the contribution of past gradients.
    vt : np.ndarray
        Array for storing the velocity (accumulated gradient), initialized to zeros.

    Methods:
    --------
    __call__(grads: np.ndarray, learning_rate: float) -> np.ndarray
        Calculates parameter updates based on current gradients and learning rate.

    reset()
        Resets the velocity term to initial values.
    """

    def __init__(self, num_params: int, momentum: float):
        """
        Initializes SGD with momentum optimizer parameters, including the velocity term
        and the momentum factor.

        Parameters:
        -----------
        num_params : int
            The number of parameters in the model that will be optimized.
        momentum : float
            The factor controlling how much of the past gradients' velocity to retain.
        """
        self.num_params = num_params  # Store the number of parameters to optimize

        # Initialize the velocity term if number of parameters > 0
        if num_params != 0:
            self.vt = np.zeros((num_params, 1))  # Velocity term (accumulated gradient)
            self.momentum = momentum  # Momentum factor

    def __call__(self, grads: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Updates the velocity term and calculates parameter update values.

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
            # Update the velocity term (vt) by scaling the previous velocity with momentum
            # and adding the current gradients
            self.vt = self.momentum * self.vt + (1 - self.momentum) * grads

            # Compute parameter updates (delta) as the velocity term
            delta = learning_rate * self.vt
            return delta  # Return computed updates for parameters

    def reset(self):
        """
        Resets the optimizer's velocity term to initial values, clearing previous
        optimization states.
        """
        # Only reset if there are parameters to reset
        if self.num_params != 0:
            self.vt = np.zeros(self.vt.shape)  # Reset velocity term to zero