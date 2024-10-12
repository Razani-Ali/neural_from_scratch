import numpy as np


class Entropy:
    def __init__(self):
        pass

    def forward(self, predictions, labels):
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
        
        # Clip values to prevent log(0)
        predictions = np.clip(predictions, 0+self.epsilon, 1-self.epsilon)
        # Calculate the binary cross-entropy loss for each sample
        loss = -np.mean(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))
        return loss

    def backward(predictions, labels):
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
        d_predictions = (predictions - labels) / (predictions * (1 - predictions) + 1e-12)
        return d_predictions

