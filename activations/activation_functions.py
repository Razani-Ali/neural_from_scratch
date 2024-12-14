import numpy as np
from activations.solid.sigmoid import sigmoid, sigmoid_derivative
from activations.solid.tanh import tanh, tanh_derivative
from activations.solid.relu import relu, relu_derivative
from activations.solid.leaky_Relu import leaky_Relu, leaky_relu_derivative
from activations.solid.elu import elu, elu_derivative
from activations.solid.softmax import softmax, softmax_derivative
from activations.solid.trigonometrics import sine, sine_derivative, cosine, cosine_derivative, sine_plus_cosine, sine_plus_cosine_derivative


def net2out(net: np.ndarray, activation_function: str, alpha: float = None) -> np.ndarray:
    """
    Applies the specified activation function to the input array.

    Parameters:
    net (np.ndarray): Input array, can be of any shape.
    layer_function (str): Name of the activation function to apply.
    alpha (float, optional): Parameter for certain activation functions (e.g., Leaky ReLU, ELU).

    Returns:
    np.ndarray: Output array after applying the specified activation function.
    """
    if activation_function == 'leaky_relu':
        if alpha is None:
            alpha = 0.01  # Default alpha for Leaky ReLU
        return leaky_Relu(net, alpha)
    elif activation_function == 'elu':
        if alpha is None:
            alpha = 1.0  # Default alpha for ELU
        return elu(net, alpha)
    elif activation_function == 'sigmoid':
        return sigmoid(net)
    elif activation_function == 'tanh':
        return tanh(net)
    elif activation_function == 'relu':
        return relu(net)
    elif activation_function == 'linear':
        return net.copy()  # Linear activation (identity function)
    elif activation_function == 'softmax':
        return softmax(net)
    elif activation_function == 'sin':
        return sine(net)
    elif activation_function == 'cos':
        return cosine(net)
    elif activation_function == 'sin+cos':
        return sine_plus_cosine(net)
    else:
        raise ValueError("The activation function derivative is not supported. supported activation functions are:"
                         "'leaky_relu', 'elu', 'sigmoid', 'tanh', 'relu', 'linear', 'softmax', 'sin', 'cos' and 'sin+cos'")
      

def net2Fprime(net: np.ndarray, activation_function: str, alpha: float = None) -> np.ndarray:
    """
    Computes the derivative of the specified activation function.

    Parameters:
    net (np.ndarray): Input array, can be of any shape.
    layer_function (str): Name of the activation function.
    alpha (float, optional): Parameter for certain activation function derivatives (e.g., Leaky ReLU, ELU).

    Returns:
    np.ndarray: Output array of derivatives for the specified activation function.
    """
    if activation_function == 'leaky_relu':
        if alpha is None:
            alpha = 0.01  # Default alpha for Leaky ReLU derivative
        return leaky_relu_derivative(net, alpha)
    elif activation_function == 'elu':
        if alpha is None:
            alpha = 1.0  # Default alpha for ELU derivative
        return elu_derivative(net, alpha)
    elif activation_function == 'sigmoid':
        return sigmoid_derivative(net)
    elif activation_function == 'tanh':
        return tanh_derivative(net)
    elif activation_function == 'relu':
        return relu_derivative(net)
    elif activation_function == 'linear':
        return np.ones(net.shape)  # Derivative of linear activation is 1
    elif activation_function == 'softmax':
        return softmax_derivative(net)
    elif activation_function == 'sin':
        return sine_derivative(net)
    elif activation_function == 'cos':
        return cosine_derivative(net)
    elif activation_function == 'sin+cos':
        return sine_plus_cosine_derivative(net)
    else:
        raise ValueError("The activation function derivative is not supported. supported activation functions are:"
                         "'leaky_relu', 'elu', 'sigmoid', 'tanh', 'relu', 'linear', 'softmax', 'sin', 'cos' and 'sin+cos'")