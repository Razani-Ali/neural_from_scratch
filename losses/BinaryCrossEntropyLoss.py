import numpy as np


class BEntropy:
    def __init__(self, eps=1e-7):
        self.eps = eps

    def forward(self, predictions: np.ndarray, labels: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the binary cross-entropy loss.
        
        :param predictions: np.ndarray of shape (batch_size,)
                            The predicted probabilities for the positive class (1).
        :param labels: np.ndarray of shape (batch_size,)
                       The true labels (0 or 1).
        :return: float
                 The average binary cross-entropy loss for the batch.
        """
        if np.ndim(predictions) == 1:
            predictions = predictions.reshape((1, len(predictions)))
            labels = labels.reshape((1, len(labels)))
        if np.ndim(labels) == 1:
            labels = labels.reshape((1, len(labels)))
        
        # Calculate the binary cross-entropy loss for each sample
        loss = -np.mean(labels * np.log(predictions + self.eps) + (1 - labels) * np.log(1 - predictions + self.eps))
        return loss

    def backward(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to the predictions.
        
        :return: np.ndarray of shape (batch_size,)
                 The gradient of the loss with respect to the predictions.
        """
        if np.ndim(predictions) == 1:
            predictions = predictions.reshape((1, len(predictions)))
            labels = labels.reshape((1, len(labels)))
        if np.ndim(labels) == 1:
            labels = labels.reshape((1, len(labels)))
        
        # Gradient of the loss with respect to the predictions
        d_predictions = (predictions - labels) / (predictions * (1 - predictions) + self.eps)
        return d_predictions

