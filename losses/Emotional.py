import numpy as np

class Emotion:
    """
    Emotion-based loss function class.

    This class calculates a loss based on error signals and rate-of-change of error
    for single-instance learning tasks (batch size of 1 only).

    Attributes
    ----------
    last_error : np.ndarray
        Stores the error from the previous forward pass.
    r_k : np.ndarray
        The computed rate-of-change of the error.
    k1 : float
        Proportional factor for the current error in the loss calculation.
    k2 : float
        Proportional factor for the change in error.
    batch_size : int, optional
        Batch size of data
    Methods
    -------
    forward(predictions: np.ndarray, labels: np.ndarray, inference: bool = False) -> float:
        Computes the forward pass, calculating the error and loss for training or evaluation.
    backward() -> np.ndarray:
        Computes the gradient of the loss with respect to the predictions.
    """

    def __init__(self, num_labels: int, k1: float = 0.8, k2: float = 0.1, batch_size: int = 32) -> None:
        """
        Initializes the Emotion loss function with given proportional constants.
        
        Parameters
        ----------
        num_labels : int
            Number of labels in the output.
        k1 : float, optional
            Weight for the current error term (default is 0.8).
        k2 : float, optional
            Weight for the change in error term (default is 0.1).
        batch_size : int, optional
            Batch size of data
        """
        self.last_error = np.zeros((1, num_labels))  # Initializes last error as zero vector
        self.r_k = np.zeros((batch_size, num_labels))  # Initializes rate of error change as zero matrix
        self.k1 = k1  # Sets proportional constant for current error
        self.k2 = k2  # Sets proportional constant for change in error
        self.batch_size = batch_size # Sets data batch size
        self.memory = True # Sets an attribute to specify whether if it is memory based or not

    def forward(self, predictions: np.ndarray, labels: np.ndarray, inference: bool = False) -> float:
        """
        Performs the forward pass, calculating error and optionally returning the loss.
        
        Parameters
        ----------
        predictions : np.ndarray
            Model output predictions.
        labels : np.ndarray
            True labels.
        inference : bool, optional
            Whether this is an inference pass (default is False).

        Returns
        -------
        float
            Computed mean loss if inference is True, otherwise None.
        """
        # Handling incosistent shapes
        if predictions.shape != labels.shape:
            raise ValueError('input arguments must have same shape, you may need to reshape labels')

        # Check that batch size is valid
        if predictions.shape[0] > self.batch_size:
            if not inference:
                raise ValueError('Data batch size cannot be larger than Loss batch size')
        
        # Store data batch size to prevent further errors
        self.data_batch_size = predictions.shape[0]

        # If not inference, calculate error and update rate of error change (r_k)
        if not inference:
            # Inference mode: calculate the loss
            loss = np.zeros((predictions.shape[0], 1))  # Initialize loss
            for index in range(labels.shape[0]):
                e = labels[index] - predictions[index]  # Error for current instance
                self.r_k[index] = self.k1 * e + self.k2 * (e - self.last_error)  # Rate of change of error
                self.last_error = e  # Update last error
                loss[index] = np.mean(0.5 * self.r_k[index] ** 2)  # Calculate squared error and store in loss
            return np.mean(loss)  # Return mean loss
        else:
            # Inference mode: calculate the loss
            last_error = np.zeros((1, labels.shape[1]))  # Initialize error for first instance
            loss = np.zeros((predictions.shape[0], 1))  # Initialize loss
            for index in range(labels.shape[0]):
                e = labels[index] - predictions[index]  # Error for current instance
                rk = self.k1 * e + self.k2 * (e - last_error)  # Rate of change of error
                last_error = e  # Update last error
                loss[index] = np.mean(0.5 * rk ** 2)  # Calculate squared error and store in loss
            return np.mean(loss)  # Return mean loss

    def backward(self) -> np.ndarray:
        """
        Performs the backward pass, calculating the gradient of loss with respect to predictions.

        Returns
        -------
        np.ndarray
            Gradient of loss with respect to predictions.
        """
        d_predictions = -(self.k1 + self.k2) * self.r_k[:self.data_batch_size]   # Gradient with respect to predictions
        return d_predictions  # Return the gradient


class Emotion2:
    """
    Second variant of the emotion-based loss function class with an additional error term.

    This class includes an extra term for the error rate-of-change over a longer timespan.

    Attributes
    ----------
    last_error : np.ndarray
        Stores the error from the previous forward pass.
    per2_error : np.ndarray
        Stores the error from two steps back.
    r_k : np.ndarray
        The computed rate-of-change of the error.
    k1 : float
        Proportional factor for the current error in the loss calculation.
    k2 : float
        Proportional factor for the change in error.
    k3 : float
        Proportional factor for the second-order change in error.
    batch_size : int, optional
        Batch size of data
    Methods
    -------
    forward(predictions: np.ndarray, labels: np.ndarray, inference: bool = False) -> float:
        Computes the forward pass, calculating the error and loss for training or evaluation.
    backward() -> np.ndarray:
        Computes the gradient of the loss with respect to the predictions.
    """

    def __init__(self, num_labels: int, k1: float = 0.8, k2: float = 0.1, k3: float = 0.1, batch_size: int = 32) -> None:
        """
        Initializes the Emotion2 loss function with given proportional constants.
        
        Parameters
        ----------
        num_labels : int
            Number of labels in the output.
        k1 : float, optional
            Weight for the current error term (default is 0.8).
        k2 : float, optional
            Weight for the change in error term (default is 0.1).
        k3 : float, optional
            Weight for the second-order change in error term (default is 0.1).
        batch_size : int, optional
            Batch size of data
        """
        self.last_error = np.zeros((1, num_labels))  # Initializes last error as zero vector
        self.per2_error = np.zeros((1, num_labels))  # Initializes older error as zero vector
        self.r_k = np.zeros((batch_size, num_labels))  # Initializes rate of error change as zero vector
        self.k1 = k1  # Sets proportional constant for current error
        self.k2 = k2  # Sets proportional constant for change in error
        self.k3 = k3  # Sets proportional constant for second-order change in error
        self.batch_size = batch_size # Sets data batch size
        self.memory = True # Sets an attribute to specify whether if it is memory based or not

    def forward(self, predictions: np.ndarray, labels: np.ndarray, inference: bool = False) -> float:
        """
        Performs the forward pass, calculating error and optionally returning the loss.
        
        Parameters
        ----------
        predictions : np.ndarray
            Model output predictions.
        labels : np.ndarray
            True labels.
        inference : bool, optional
            Whether this is an inference pass (default is False).

        Returns
        -------
        float
            Computed mean loss if inference is True, otherwise None.
        """
        # Handling incosistent shapes
        if predictions.shape != labels.shape:
            raise ValueError('input arguments must have same shape, you may need to reshape labels')

        # Check that batch size is 1
        if predictions.shape[0] > self.batch_size:
            if not inference:
                raise ValueError('Data batch size cannot be larger than Loss batch size')
            
        # Store data batch size to prevent further errors
        self.data_batch_size = predictions.shape[0]

        # If not inference, calculate error and update rate of error change (r_k)
        if not inference:
            loss = np.zeros((predictions.shape[0], 1))  # Initialize loss
            for index in range(labels.shape[0]):
                e = labels[index] - predictions[index]  # Error for current instance
                self.r_k[index] = (self.k1 + self.k2 + self.k3) * e - \
                    (self.k1 + self.k2) * self.last_error + self.k3 * self.per2_error  # Rate of change of error
                self.per2_error = self.last_error.copy()  # Update two-step-back error
                self.last_error = e.copy()  # Update last error
                loss[index] = np.mean(0.5 * self.r_k[index] ** 2)  # Calculate squared error and store in loss
            return np.mean(loss)  # Return mean loss
        else:
            # Inference mode: calculate the loss
            last_error = np.zeros((1, labels.shape[1]))  # Initialize error for first instance
            per2_error = np.zeros((1, labels.shape[1]))  # Initialize two-step-back error
            loss = np.zeros((predictions.shape[0], 1))  # Initialize loss
            for index in range(labels.shape[0]):
                e = labels[index] - predictions[index]  # Error for current instance
                rk = self.k1 * e + self.k2 * (e - last_error) + self.k3 * (last_error - per2_error)  # Rate of change of error
                per2_error = last_error.copy()  # Update two-step-back error
                last_error = e.copy()  # Update last error
                loss[index] = np.mean(0.5 * rk ** 2)  # Calculate squared error and store in loss
            return np.mean(loss)  # Return mean loss

    def backward(self) -> np.ndarray:
        """
        Performs the backward pass, calculating the gradient of loss with respect to predictions.

        Returns
        -------
        np.ndarray
            Gradient of loss with respect to predictions.
        """
        d_predictions = -(self.k1 + self.k2 + self.k3) * self.r_k[:self.data_batch_size]  # Gradient with respect to predictions
        return d_predictions  # Return the gradient
