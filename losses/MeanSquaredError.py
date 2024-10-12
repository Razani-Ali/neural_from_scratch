import numpy as np


class MSE:
    def __init__(self):
        pass

    def forward(predictions, labels):
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
        if np.ndim(predictions) == 1:
            predictions = predictions.reshape((1, len(predictions)))
            labels = labels.reshape((1, len(labels)))
        if np.ndim(labels) == 1:
            labels = labels.reshape((1, len(labels)))
        
        # Calculate the mean squared error loss
        loss = 0.5 * np.mean((predictions - labels) ** 2, axis=1)
        return np.mean(loss)

    def backward(predictions, labels):
        """
        Compute the gradient of the loss with respect to the predictions.
        
        :return: np.ndarray of shape (batch_size, num_features)
                 The gradient of the loss with respect to the predictions.
        """
        # Handling incosistent shapes
        if np.ndim(predictions) == 1:
            predictions = predictions.reshape((1, len(predictions)))
            labels = labels.reshape((1, len(labels)))
        if np.ndim(labels) == 1:
            labels = labels.reshape((1, len(labels)))
        
        # Gradient of the loss with respect to the predictions
        d_predictions = predictions - labels
        return d_predictions