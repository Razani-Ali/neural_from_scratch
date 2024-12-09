import numpy as np


class MAE:
    def __init__(self):
        pass

    def forward(self, predictions, labels, **kwargs):
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
        if np.ndim(predictions) == 1:
            predictions = predictions.reshape((1, len(predictions)))
            labels = labels.reshape((1, len(labels)))
        if np.ndim(labels) == 1:
            labels = labels.reshape((1, len(labels)))

        # Calculate the mean absolute error loss
        loss = np.abs(predictions - labels, axis=1)
        return np.mean(loss)

    def backward(self, predictions, labels):
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
        d_predictions = np.sign(predictions - labels)
        return d_predictions
