import numpy as np


class MSE:
    def __init__(self):
        pass

    def forward(self, predictions: np.ndarray, labels: np.ndarray, **kwargs) -> np.float64:
        """
        Compute the Mean Squared Error (MSE) loss.
        
        :param predictions: np.ndarray of shape (batch_size, num_features)
                            The predicted values.
        :param labels: np.ndarray of shape (batch_size, num_features)
                       The true values.
        :return: float
                 The average MSE loss for the batch.
        """
        # Handling incosistent shapes
        if predictions.shape != labels.shape:
            raise ValueError('input arguments must have same shape, you may need to reshape labels')
        
        # Calculate the mean squared error loss
        loss = 0.5 * np.mean(np.square(predictions - labels))
        return loss

    def backward(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to the predictions.
        
        :return: np.ndarray of shape (batch_size, num_features)
                 The gradient of the loss with respect to the predictions.
        """
        # Handling incosistent shapes
        if predictions.shape != labels.shape:
            raise ValueError('input arguments must have same shape, you may need to reshape labels')
        
        # Gradient of the loss with respect to the predictions
        d_predictions = predictions - labels
        return d_predictions