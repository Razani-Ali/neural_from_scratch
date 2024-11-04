import numpy as np

class Adam:
    """
    Custom implementation of the Adam optimizer for gradient-based optimization.
    Stores the state of first and second moment estimates for each parameter and 
    updates them based on the gradients provided.

    Attributes:
    -----------
    num_params : int
        Number of parameters to be optimized.
    beta1 : float
        Exponential decay rate for the first moment estimate.
    beta2 : float
        Exponential decay rate for the second moment estimate.
    eps : float
        Small constant to prevent division by zero in update rule.
    mt : np.ndarray
        Array for first moment estimates, initialized to zeros.
    vt : np.ndarray
        Array for second moment estimates, initialized to zeros.
    t : int
        Time step counter for computing bias-corrected estimates.
    
    Methods:
    --------
    __call__(grads: np.ndarray, learning_rate: float) -> np.ndarray
        Calculates parameter updates based on current gradients and learning rate.
    
    reset()
        Resets first and second moment estimates and time step counter to initial values.
    """

    def __init__(self, num_params: int, beta1: float, beta2: float, eps: float):
        """
        Initializes Adam optimizer parameters, including moment estimates, time step, 
        and decay rates.

        Parameters:
        -----------
        num_params : int
            The number of parameters in the model that will be optimized.
        beta1 : float
            The decay rate for the first moment estimate (mean of gradients).
        beta2 : float
            The decay rate for the second moment estimate (uncentered variance of gradients).
        eps : float
            A small constant added to avoid division by zero.
        """
        self.num_params = num_params  # Store number of parameters to optimize
        
        # Initialize first and second moment estimates if number of parameters > 0
        if num_params != 0:
            self.mt = np.zeros((num_params, 1))  # First moment estimate (mean of gradients)
            self.vt = np.zeros((num_params, 1))  # Second moment estimate (variance of gradients)
            self.t = 0  # Time step counter initialized to zero
            self.beta1 = beta1  # Decay rate for the first moment
            self.beta2 = beta2  # Decay rate for the second moment
            self.eps = eps  # Small constant for numerical stability

    def __call__(self, grads: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Updates the optimizer's first and second moment estimates and calculates 
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
            
            # Update biased second moment estimate (vt) with exponential decay
            self.vt = self.beta2 * self.vt + (1 - self.beta2) * np.square(grads)
            
            # Correct bias in first moment estimate
            m_hat = self.mt / (1 - self.beta1 ** self.t)
            
            # Correct bias in second moment estimate
            v_hat = self.vt / (1 - self.beta2 ** self.t)
            
            # Compute parameter updates (delta) with corrected moments and learning rate
            delta = learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
            return delta  # Return computed updates for parameters
    
    def reset(self):
        """
        Resets the optimizer's moment estimates and time step counter to initial values,
        clearing previous optimization states.
        """
        # Only reset if there are parameters to reset
        if self.num_params != 0:
            self.mt = np.zeros(self.mt.shape)  # Reset first moment estimate to zero
            self.vt = np.zeros(self.vt.shape)  # Reset second moment estimate to zero
            self.t = 0  # Reset time step counter
