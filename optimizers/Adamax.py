import numpy as np


class Adamax:
    """
    Custom implementation of the Adamax optimizer, a variant of Adam based on the
    infinity norm (maximum absolute value of gradients) for gradient-based optimization.
    This optimizer is well-suited for models with sparse gradients and can handle
    very large gradients effectively.

    Attributes:
    -----------
    num_params : int
        Number of parameters to be optimized.
    beta1 : float
        Exponential decay rate for the first moment estimate.
    beta2 : float
        Exponential decay rate for the second moment (infinity norm) estimate.
    eps : float
        Small constant to prevent division by zero in the update rule.
    mt : np.ndarray
        Array for first moment estimates, initialized to zeros.
    ut : np.ndarray
        Array for infinity norm estimates of gradients, initialized to zeros.
    t : int
        Time step counter for computing bias-corrected estimates.

    Methods:
    --------
    __call__(grads: np.ndarray, learning_rate: float) -> np.ndarray
        Calculates parameter updates based on current gradients and learning rate.

    reset()
        Resets first and infinity norm estimates and time step counter to initial values.
    """

    def __init__(self, num_params: int, beta1: float, beta2: float, eps: float):
        """
        Initializes Adamax optimizer parameters, including moment estimates,
        time step, and decay rates.

        Parameters:
        -----------
        num_params : int
            The number of parameters in the model that will be optimized.
        beta1 : float
            The decay rate for the first moment estimate (mean of gradients).
        beta2 : float
            The decay rate for the infinity norm estimate of gradients.
        eps : float
            A small constant added to avoid division by zero.
        """
        self.num_params = num_params  # Store number of parameters to optimize

        # Initialize first and infinity norm estimates if number of parameters > 0
        if num_params != 0:
            self.mt = np.zeros((num_params, 1))  # First moment estimate (mean of gradients)
            self.ut = np.zeros((num_params, 1))  # Infinity norm estimate (max abs gradient)
            self.t = 0  # Time step counter initialized to zero
            self.beta1 = beta1  # Decay rate for the first moment
            self.beta2 = beta2  # Decay rate for the infinity norm
            self.eps = eps  # Small constant for numerical stability

    def __call__(self, grads: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Updates the optimizer's first moment and infinity norm estimates, and calculates
        parameter update values.

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
            self.t += 1  # Increment time step counter

            # Update biased first moment estimate (mt) with exponential decay
            self.mt = self.beta1 * self.mt + (1 - self.beta1) * grads

            # Update biased infinity norm estimate (ut) with exponential decay
            self.ut = np.maximum(self.beta2 * self.ut, np.abs(grads))

            # Correct bias in first moment estimate
            m_hat = self.mt / (1 - self.beta1 ** self.t)

            # Compute parameter updates (delta) using m_hat and infinity norm estimate
            delta = learning_rate * m_hat / (self.ut + self.eps)
            return delta  # Return computed updates for parameters

    def reset(self):
        """
        Resets the optimizer's moment estimates and time step counter to initial values,
        clearing previous optimization states.
        """
        # Only reset if there are parameters to reset
        if self.num_params != 0:
            self.mt = np.zeros(self.mt.shape)  # Reset first moment estimate to zero
            self.ut = np.zeros(self.ut.shape)  # Reset infinity norm estimate to zero
            self.t = 0  # Reset time step counter