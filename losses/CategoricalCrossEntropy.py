import numpy as np


class Entropy:
    def __init__(self):
        pass

    def forward(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute the categorical cross-entropy loss.
        
        :param predictions: np.ndarray of shape (batch_size, num_classes)
                            The predicted probabilities for each class.
        :param labels: np.ndarray of shape (batch_size, num_classes)
                       The one-hot encoded true labels.
        :return: float
                 The average categorical cross-entropy loss for the batch.
        """

        # Calculating Loss
        E = np.sum(-labels * np.log(predictions), axis=1)
        return np.mean(E)

    def backward(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
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
        
        # Gradient of the loss with respect to the predictions
        return -(labels / (predictions + 1e-12))