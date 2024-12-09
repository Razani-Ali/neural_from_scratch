import numpy as np


def flex_sig(a: float, net: float) -> float:
    """
    Computes the flexible sigmoid activation function.

    Parameters:
    a (float): Flexibility factor that modulates the steepness of the sigmoid.
    net (float): Input value to the sigmoid function (net input to the neuron).

    Returns:
    float: The result of the flexible sigmoid function.
    """
    # Step 1: Compute the flexible sigmoid function
    # The sigmoid is scaled by the absolute value of 'a' and stretched
    # or compressed by the parameter 'net'.
    x = np.clip(-net * a, -709.0, 709.0)  # To prevent numerical issues
    return 2 * np.abs(a) / (1 + np.exp(x))


def flex_sig_star_derivative(a: float, net: float) -> float:
    """
    Computes the derivative of the flexible sigmoid function with respect to 'a', 
    the parameter that modulates the flexibility of the sigmoid.

    Parameters:
    a (float): Flexibility factor that modulates the steepness of the sigmoid.
    net (float): Input value to the sigmoid function (net input to the neuron).

    Returns:
    float: The derivative of the flexible sigmoid function with respect to 'a'.
    """
    # Step 1: Calculate the standard sigmoid value 'g' using 'net * a'
    x = np.clip(-net * a, -709.0, 709.0)  # To prevent numerical issues
    g = 1 / (1 + np.exp(x))
    
    # Step 2: Calculate the sign of 'a' (positive, negative, or zero)
    sign_a = np.sign(a)
    
    # Step 3: Calculate the derivative with respect to 'a'
    # The derivative consists of two terms: 
    # 1. A sign-based scaling of the sigmoid.
    # 2. A second term that takes into account both 'a' and 'net'.
    df_da = 2 * (sign_a * g + np.abs(a) * net * g * (1 - g))
    
    # Return the computed derivative
    return df_da


def flex_sig_derivative(a: float, net: float) -> float:
    """
    Computes the derivative of the flexible sigmoid function with respect to 'net', 
    the input to the neuron.

    Parameters:
    a (float): Flexibility factor that modulates the steepness of the sigmoid.
    net (float): Input value to the sigmoid function (net input to the neuron).

    Returns:
    float: The derivative of the flexible sigmoid function with respect to 'net'.
    """
    # Step 1: Calculate the standard sigmoid value 'g' using 'net * a'
    x = np.clip(-net * a, -709.0, 709.0)  # To prevent numerical issues
    g = 1 / (1 + np.exp(x))
    
    # Step 2: Compute the derivative with respect to 'net'
    # This derivative includes the absolute value of 'a' and a sigmoid factor.
    df_dnet = 2 * np.abs(a) * a * g * (1 - g)
    
    # Return the computed derivative
    return df_dnet
