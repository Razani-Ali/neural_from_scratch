from optimizers.Adam import Adam

def init_optimizer(num_params: int, method: str = 'Adam', **kwargs) -> Adam:
    """
    Initializes and returns the specified optimizer for updating model parameters.
    
    Parameters:
    -----------
    num_params : int
        The number of parameters in the model to be optimized.
    method : str, optional
        The optimizer to use, default is 'Adam'.
    **kwargs : dict, optional
        Additional keyword arguments to customize the optimizer's settings,
        particularly for Adam, including 'beta1', 'beta2', and 'eps'.
        
    Returns:
    --------
    Adam
        An instance of the Adam optimizer configured with the given parameters.
        
    Raises:
    -------
    ValueError
        If an unsupported optimizer method is specified.
    """
    
    # Check if the selected optimization method is Adam
    if method == 'Adam':
        # Set beta1, the decay rate for the first moment estimate, with a default of 0.9
        beta1 = kwargs.get('beta1', 0.9)
        
        # Set beta2, the decay rate for the second moment estimate, with a default of 0.99
        beta2 = kwargs.get('beta2', 0.99)
        
        # Set eps, a small constant to prevent division by zero, with a default of 1e-7
        eps = kwargs.get('eps', 1e-7)
        
        # Initialize and return an instance of the Adam optimizer with the specified parameters
        return Adam(num_params, beta1, beta2, eps)
    
    else:
        # Raise an error if the specified optimizer method is not supported
        raise ValueError('your optimizer is not supported')
