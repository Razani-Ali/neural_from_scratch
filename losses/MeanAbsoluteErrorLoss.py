import numpy as np


class MAE:
    def __init__(self):
        pass

    def forward(self, predictions: np.ndarray, labels: np.ndarray, **kwargs) -> np.float64:
        """
        Compute the Mean Absolute Error (MAE) loss.
        
        :param predictions: np.ndarray of shape (batch_size, num_features)
                            The predicted values.
        :param labels: np.ndarray of shape (batch_size, num_features)
                       The true values.
        :return: float
                 The average MAE loss for the batch.
        """
        # Handling incosistent shapes
        if predictions.shape != labels.shape:
            raise ValueError('input arguments must have same shape, you may need to reshape labels')

        # Calculate the mean absolute error loss
        loss = np.abs(predictions - labels)
        return np.mean(loss)

    def backward(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to the predictions.
        
        :return: np.ndarray of shape (batch_size, num_features)
                 The gradient of the loss with respect to the predictions.
        """
        if predictions.shape != labels.shape:
            raise ValueError('input arguments must have same shape, you may need to reshape labels')
        
        # Gradient of the loss with respect to the predictions
        d_predictions = np.sign(predictions - labels)
        return d_predictions
