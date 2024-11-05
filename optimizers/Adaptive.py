import numpy as np

class AdaptiveSGD:
    """
    Adaptive Stochastic Gradient Descent (AdaptiveSGD) optimizer.
    
    This optimizer adapts the learning rate for each parameter based on the gradient history.
    
    Attributes
    ----------
    num_params : int
        The number of parameters in the model.
    learning_rate_eta : float
        A factor that controls the rate of adjustment for the learning rate based on past gradients.
    last_grad : np.ndarray
        Stores the gradient from the previous update step, used to adjust the learning rate adaptively.
        
    Methods
    -------
    __call__(grads: np.ndarray, learning_rate: float) -> np.ndarray:
        Calculates the parameter update using the current gradients and adjusts the learning rate.
    reset() -> None:
        Resets the optimizer's internal states (last gradient and learning rate array).
    """

    def __init__(self, num_params: int, learning_rate_eta: float, eta_up: float, eta_low: float) -> None:
        """
        Initializes the AdaptiveSGD optimizer with the given number of parameters and 
        learning rate adjustment factor.

        Parameters
        ----------
        num_params : int
            The number of parameters in the model that will be optimized.
        learning_rate_eta : float
            The factor for adapting the learning rate based on gradient history.
        eta_up
            Upper bound of learning rates
        eta_low
            Lower bound of learning rates
        """
        self.num_params = num_params  # Store the number of parameters
        self.learning_rate_eta = learning_rate_eta  # Set the learning rate adjustment factor
        self.upper_bound = eta_up  # Set the upper bound for learning_rate attribute
        self.lower_bound = eta_low  # Set the lower bound for learning_rate attribute

        # Initialize last_grad as a zero vector if num_params is not zero
        if num_params != 0:
            self.last_grad = np.zeros((num_params, 1))  # Last gradient placeholder

    def __call__(self, grads: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Computes the parameter update step using adaptive learning rates based on gradient history.
        
        Parameters
        ----------
        grads : np.ndarray
            The current gradient of the model's parameters.
        learning_rate : float
            The initial learning rate for this step.

        Returns
        -------
        np.ndarray
            The computed update (delta) to be applied to the model's parameters.
        """
        # Ensure that gradients are provided
        if grads is not None:
            # Check if the learning rate array is initialized; if not, initialize it
            if not hasattr(self, 'learning_rate'):
                # Initialize learning_rate as an array of the specified learning_rate value
                self.learning_rate = np.full((self.num_params, 1), learning_rate)

            # Update the learning rate for each parameter adaptively based on the previous gradient
            self.learning_rate -= self.learning_rate_eta * grads * self.last_grad
            self.learning_rate = np.clip(self.learning_rate, self.lower_bound, self.upper_bound)
            
            # Update last_grad to hold the current gradient for the next step
            self.last_grad = grads
            
            # Calculate the parameter update delta as the product of learning_rate and grads
            delta = self.learning_rate * grads
            
            return delta  # Return the computed parameter update step

    def reset(self) -> None:
        """
        Resets the internal states of the optimizer, including the last gradient and learning rate array.
        """
        # Reset last_grad to a zero vector if there are parameters to track
        if self.num_params != 0:
            self.last_grad = np.zeros((self.num_params, 1))  # Reset last gradient
            del self.learning_rate  # Remove the current learning rate array to allow reinitialization
