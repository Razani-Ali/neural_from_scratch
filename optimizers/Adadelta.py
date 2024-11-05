import numpy as np


class Adadelta:
    """
    Custom implementation of the Adadelta optimizer for gradient-based optimization.
    Adadelta adapts learning rates based on a moving window of gradient updates,
    addressing the diminishing learning rates observed in Adagrad.

    Attributes:
    -----------
    num_params : int
        Number of parameters to be optimized.
    rho : float
        Decay rate for moving average of squared gradients.
    eps : float
        Small constant to prevent division by zero in the update rule.
    Eg2 : np.ndarray
        Array for the moving average of squared gradients, initialized to zeros.
    Ex2 : np.ndarray
        Array for the moving average of squared parameter updates, initialized to zeros.

    Methods:
    --------
    __call__(grads: np.ndarray, learning_rate: float) -> np.ndarray
        Calculates parameter updates based on current gradients.

    reset()
        Resets the moving averages of squared gradients and updates to initial values.
    """

    def __init__(self, num_params: int, rho: float, eps: float):
        """
        Initializes Adadelta optimizer parameters, including the moving averages
        of squared gradients and updates, and decay rate.

        Parameters:
        -----------
        num_params : int
            The number of parameters in the model that will be optimized.
        rho : float
            The decay rate for moving averages.
        eps : float
            A small constant added to avoid division by zero.
        """
        self.num_params = num_params  # Store the number of parameters to optimize

        # Initialize the moving averages if number of parameters > 0
        if num_params != 0:
            self.Eg2 = np.zeros((num_params, 1))  # Moving average of squared gradients
            self.Ex2 = np.zeros((num_params, 1))  # Moving average of squared parameter updates
            self.rho = rho  # Decay rate for the moving averages
            self.eps = eps  # Small constant for numerical stability

    def __call__(self, grads: np.ndarray, learning_rate: float = 1.0) -> np.ndarray:
        """
        Updates the moving averages of squared gradients and squared updates, and
        calculates parameter update values.

        Parameters:
        -----------
        grads : np.ndarray
            The gradient values for each parameter (size matches num_params).
        learning_rate : float, optional
            The learning rate is typically set to 1.0 in Adadelta, but can be adjusted.

        Returns:
        --------
        delta : np.ndarray
            Array of parameter update values.
        """
        if grads is not None:
            # Update the moving average of squared gradients (Eg2)
            self.Eg2 = self.rho * self.Eg2 + (1 - self.rho) * np.square(grads)

            # Compute the adaptive learning rate for each parameter
            rms_delta = np.sqrt(self.Ex2 + self.eps)  # RMS of previous updates
            rms_g = np.sqrt(self.Eg2 + self.eps)      # RMS of current gradients
            delta = (rms_delta / rms_g) * grads      # Compute parameter update

            # Update the moving average of squared updates (Ex2) based on delta
            self.Ex2 = self.rho * self.Ex2 + (1 - self.rho) * np.square(delta)

            return delta  # Return computed updates for parameters

    def reset(self):
        """
        Resets the optimizer's moving averages of squared gradients and updates
        to initial values, clearing previous optimization states.
        """
        # Only reset if there are parameters to reset
        if self.num_params != 0:
            self.Eg2 = np.zeros(self.Eg2.shape)  # Reset moving average of squared gradients
            self.Ex2 = np.zeros(self.Ex2.shape)  # Reset moving average of squared updates