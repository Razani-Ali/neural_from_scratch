from optimizers.Adam import Adam
from optimizers.Adadelta import Adadelta
from optimizers.Adagrad import Adagrad
from optimizers.Adamax import Adamax
from optimizers.Adaptive import AdaptiveSGD
from optimizers.AMSgrad import AMSGrad
from optimizers.Momentum import Momentum
from optimizers.Nadam import Nadam
from optimizers.RMSprop import RMSprop
from optimizers.SGD import SGD

def init_optimizer(num_params: int, method: str = 'Adam', **kwargs):
    """
    Initializes and returns the specified optimizer for updating model parameters.
    
    Parameters
    ----------
    num_params : int
        The number of parameters in the model to be optimized.
    method : str, optional
        The optimizer to use, default is 'Adam'. Supported options are 'Adam', 'Adadelta', 
        'Adagrad', 'Adamax', 'AdaptiveSGD', 'AMSGrad', 'Momentum', 'Nadam', 'RMSprop', and 'SGD'.
    **kwargs : dict, optional
        Additional keyword arguments to customize each optimizer’s settings.
        
        - For Adam, Adamax, AMSGrad, Nadam:
          `beta1` (float): decay rate for first moment; `beta2` (float): decay rate for second moment; `eps` (float): small constant.
          
        - For Adadelta:
          `rho` (float): decay rate for moving averages; `eps` (float): small constant.
          
        - For Adagrad:
          `eps` (float): small constant.
          
        - For AdaptiveSGD:
          `learning_rate_eta` (float): learning rate factor.
          
        - For Momentum:
          `momentum` (float): factor controlling past gradient velocity.
          
        - For RMSprop:
          `beta` (float): decay rate for squared gradients; `eps` (float): small constant.
          
    Returns
    -------
    Optimizer
        An instance of the chosen optimizer configured with the specified parameters.
        
    Raises
    ------
    ValueError
        If an unsupported optimizer method is specified.
    """
    
    if method == 'Adam':
        # Initialize Adam with beta1, beta2, and eps as additional parameters
        beta1 = kwargs.get('beta1', 0.9)
        beta2 = kwargs.get('beta2', 0.99)
        eps = kwargs.get('eps', 1e-7)
        return Adam(num_params, beta1, beta2, eps)
    
    elif method == 'Adadelta':
        # Initialize Adadelta with rho and eps as additional parameters
        rho = kwargs.get('rho', 0.95)
        eps = kwargs.get('eps', 1e-6)
        return Adadelta(num_params, rho, eps)
    
    elif method == 'Adagrad':
        # Initialize Adagrad with eps as an additional parameter
        eps = kwargs.get('eps', 1e-7)
        return Adagrad(num_params, eps)
    
    elif method == 'Adamax':
        # Initialize Adamax with beta1, beta2, and eps as additional parameters
        beta1 = kwargs.get('beta1', 0.9)
        beta2 = kwargs.get('beta2', 0.999)
        eps = kwargs.get('eps', 1e-7)
        return Adamax(num_params, beta1, beta2, eps)
    
    elif method == 'AdaptiveSGD':
        # Initialize AdaptiveSGD with a learning rate factor as an additional parameter
        learning_rate_eta = kwargs.get('learning_rate_eta', 1e-3)
        eta_up = kwargs.get('eta_upper_bound', 1.0)
        eta_low = kwargs.get('eta_lower_bound', 1e-6)
        return AdaptiveSGD(num_params, learning_rate_eta, eta_up, eta_low)
    
    elif method == 'AMSGrad':
        # Initialize AMSGrad with beta1, beta2, and eps as additional parameters
        beta1 = kwargs.get('beta1', 0.9)
        beta2 = kwargs.get('beta2', 0.999)
        eps = kwargs.get('eps', 1e-7)
        return AMSGrad(num_params, beta1, beta2, eps)
    
    elif method == 'Momentum':
        # Initialize Momentum with a momentum factor as an additional parameter
        momentum = kwargs.get('momentum', 0.9)
        return Momentum(num_params, momentum)
    
    elif method == 'Nadam':
        # Initialize Nadam with beta1, beta2, and eps as additional parameters
        beta1 = kwargs.get('beta1', 0.9)
        beta2 = kwargs.get('beta2', 0.999)
        eps = kwargs.get('eps', 1e-7)
        return Nadam(num_params, beta1, beta2, eps)
    
    elif method == 'RMSprop':
        # Initialize RMSprop with beta and eps as additional parameters
        beta = kwargs.get('beta', 0.9)
        eps = kwargs.get('eps', 1e-7)
        return RMSprop(num_params, beta, eps)
    
    elif method == 'SGD':
        # Initialize SGD without any additional parameters
        return SGD(num_params)
    
    else:
        # Raise an error if an unsupported optimizer method is specified
        raise ValueError('Unsupported optimizer method specified. Available options are: '
                         'Adam, Adadelta, Adagrad, Adamax, AdaptiveSGD, AMSGrad, Momentum, '
                         'Nadam, RMSprop, SGD')
