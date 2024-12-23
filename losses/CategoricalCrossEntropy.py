import numpy as np


class Entropy:
    def __init__(self, eps=1e-7):
        self.eps = eps

    def forward(self, predictions: np.ndarray, labels: np.ndarray, **kwargs) -> np.float64:
        """
        Compute the categorical cross-entropy loss.
        
        :param predictions: np.ndarray of shape (batch_size, num_classes)
                            The predicted probabilities for each class.
        :param labels: np.ndarray of shape (batch_size, num_classes)
                       The one-hot encoded true labels.
        :return: float
                 The average categorical cross-entropy loss for the batch.
        """
        # Handling incosistent shapes
        if predictions.shape != labels.shape:
            raise ValueError('input arguments must have same shape, you may need to reshape labels')
        
        # Calculating Loss
        E = np.sum(-labels * np.log(predictions + self.eps), axis=1)
        return np.mean(E)

    def backward(self, predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to the predictions.
        
        :param predictions: np.ndarray of shape (batch_size, num_classes)
                            The predicted probabilities for each class.
        :param labels: np.ndarray of shape (batch_size, num_classes)
                       The one-hot encoded true labels.

        :return: np.ndarray of shape (batch_size, num_classes)
                 The gradient of the loss with respect to the predictions.
        """
        # Handling incosistent shapes
        if predictions.shape != labels.shape:
            raise ValueError('input arguments must have same shape, you may need to reshape labels')
        
        # Gradient of the loss with respect to the predictions
        return -(labels / (predictions + self.eps))