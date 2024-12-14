import numpy as np
from activations.flexible.flexible_sigmoid import flex_sig, flex_sig_derivative, flex_sig_star_derivative
from activations.flexible.flexible_tanh import flex_tanh, flex_tanh_derivative, flex_tanh_star_derivative
from activations.flexible.flexible_leaky_relu import flex_leaky_relu, flex_leaky_relu_derivative
from activations.flexible.flexible_leaky_relu import flex_leaky_relu_star_derivative
from activations.flexible.selu import selu, selu_derivative, selu_star_derivative, selu_lambda_derivative
from activations.flexible.flexible_sin import sin_flexible, sin_flexible_alpha_derivative,\
    sin_flexible_derivative, sin_flexible_lambda_derivative
from activations.flexible.flexible_cos import cos_flexible, cos_flexible_alpha_derivative,\
    cos_flexible_derivative, cos_flexible_lambda_derivative
from activations.flexible.flexible_sincos import sin_plus_cos_flexible, sin_plus_cos_flexible_alpha_derivative,\
    sin_plus_cos_flexible_derivative, sin_plus_cos_flexible_lambda_derivative


def net2out(net: np.ndarray, activation_function: str, alpha: float, lambda_param: float = None) -> np.ndarray:
    """
    Applies the specified activation function to the input 'net'.

    Parameters:
    - net (np.ndarray): Input array (neuron activations).
    - activation_function (str): Type of activation function ('leaky_relu', 'selu', 'elu', 'sigmoid', 'tanh').
    - alpha (float): Parameter for activation functions like 'leaky_relu', 'selu', etc.
    - lambda_param (float, optional): Scaling factor used for 'selu' and 'elu' functions.

    Returns:
    - np.ndarray: Output after applying the chosen activation function.
    """
    # Leaky ReLU activation
    if activation_function == 'leaky_relu':
        return flex_leaky_relu(net, alpha)
    # SELU or ELU activation
    elif activation_function == 'selu' or activation_function == 'elu':
        return selu(net, lambda_param, alpha)
    # Sigmoid activation
    elif activation_function == 'sigmoid':
        return flex_sig(alpha, net)
    # Tanh activation
    elif activation_function == 'tanh':
        return flex_tanh(alpha, net)
    # Sin activation
    elif activation_function == 'sin':
        return sin_flexible(net, alpha, lambda_param)
    # Cos activation
    elif activation_function == 'cos':
        return sin_flexible(net, alpha, lambda_param)
    # Sin + Cos activation
    elif activation_function == 'sin+cos':
        return sin_flexible(net, alpha, lambda_param)
    elif activation_function == 'linear':
        return alpha * net
    else:
        raise ValueError("The activation function derivative is not supported. supported activation functions are:"
                         "'leaky_relu', 'elu', 'sigmoid', 'tanh', 'linear', 'sin', 'cos' and 'sin+cos'")


def net2Fprime(net: np.ndarray, activation_function: str, alpha: float, lambda_param: float = None) -> np.ndarray:
    """
    Computes the derivative of the specified activation function with respect to 'net'.

    Parameters:
    - net (np.ndarray): Input array (neuron activations).
    - activation_function (str): Type of activation function ('leaky_relu', 'selu', 'elu', 'sigmoid', 'tanh').
    - alpha (float): Parameter for activation functions like 'leaky_relu', 'selu', etc.
    - lambda_param (float, optional): Scaling factor used for 'selu' and 'elu' functions.

    Returns:
    - np.ndarray: Derivative of the activation function with respect to 'net'.
    """
    # Derivative for Leaky ReLU
    if activation_function == 'leaky_relu':
        return flex_leaky_relu_derivative(net, alpha)
    # Derivative for SELU or ELU
    elif activation_function == 'selu' or activation_function == 'elu':
        return selu_derivative(net, lambda_param, alpha)
    # Derivative for Sigmoid
    elif activation_function == 'sigmoid':
        return flex_sig_derivative(alpha, net)
    # Derivative for Tanh
    elif activation_function == 'tanh':
        return flex_tanh_derivative(alpha, net)
    elif activation_function == 'sin':
        return sin_flexible_derivative(net, alpha, lambda_param)
    # Cos activation
    elif activation_function == 'cos':
        return sin_flexible_derivative(net, alpha, lambda_param)
    # Sin + Cos activation
    elif activation_function == 'sin+cos':
        return sin_flexible_derivative(net, alpha, lambda_param)
    elif activation_function == 'linear':
        return alpha
    else:
        raise ValueError("The activation function derivative is not supported. supported activation functions are:"
                         "'leaky_relu', 'elu', 'sigmoid', 'tanh', 'linear', 'sin', 'cos' and 'sin+cos'")


def net2Fstar(net: np.ndarray, activation_function: str, alpha: float, lambda_param: float = None):
    """
    Computes the derivative of the activation function with respect to additional parameters
    (like 'alpha' or 'lambda_param').

    Parameters:
    - net (np.ndarray): Input array (neuron activations).
    - activation_function (str): Type of activation function ('leaky_relu', 'selu', 'elu', 'sigmoid', 'tanh').
    - alpha (float): Parameter for activation functions like 'leaky_relu', 'selu', etc.
    - lambda_param (float, optional): Scaling factor used for 'selu' and 'elu' functions.

    Returns:
    - For most activations: np.ndarray (derivative w.r.t. additional parameters like alpha).
    - For 'selu'/'elu': Tuple[np.ndarray, np.ndarray] (derivatives w.r.t. 'alpha' and 'lambda_param').
    """
    # Derivative w.r.t. alpha for Leaky ReLU
    if activation_function == 'leaky_relu':
        return flex_leaky_relu_star_derivative(net, alpha)
    # Derivative w.r.t. alpha and lambda for SELU or ELU
    elif activation_function == 'selu' or activation_function == 'elu':
        return selu_star_derivative(net, lambda_param, alpha),\
            selu_lambda_derivative(net, lambda_param, alpha)
    # Derivative w.r.t. alpha for Sigmoid
    elif activation_function == 'sigmoid':
        return flex_sig_star_derivative(alpha, net)
    # Derivative w.r.t. alpha for Tanh
    elif activation_function == 'tanh':
        return flex_tanh_star_derivative(alpha, net)
    elif activation_function == 'sin':
        return sin_flexible_alpha_derivative(net, alpha, lambda_param),\
            sin_flexible_lambda_derivative(net, alpha, lambda_param)
    # Cos activation
    elif activation_function == 'cos':
        return cos_flexible_alpha_derivative(net, alpha, lambda_param),\
            cos_flexible_lambda_derivative(net, alpha, lambda_param)
    # Sin + Cos activation
    elif activation_function == 'sin+cos':
        return sin_plus_cos_flexible_alpha_derivative(net, alpha, lambda_param),\
            sin_plus_cos_flexible_lambda_derivative(net, alpha, lambda_param)
    elif activation_function == 'linear':
        return net
    else:
        raise ValueError("The activation function derivative is not supported. supported activation functions are:"
                         "'leaky_relu', 'elu', 'sigmoid', 'tanh', 'linear', 'sin', 'cos' and 'sin+cos'")
