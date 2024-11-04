from optimizers.Adam import Adam


def init_optimizer(num_params, method = 'Adam', **kwargs):
    if method == 'Adam':
        beta1 = kwargs.get('beta1', 0.9)
        beta2 = kwargs.get('beta2', 0.99)
        eps = kwargs.get('eps', 1e-7)
        return Adam(num_params, beta1, beta2, eps)
    else:
        raise ValueError('your optimizer is not supported')
