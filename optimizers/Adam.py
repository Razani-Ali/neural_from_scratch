import numpy as np


class Adam:
    def __init__(self, num_params, beta1, beta2, eps):
        self.num_params = num_params
        if num_params != 0:
            self.mt = np.zeros((num_params, 1))
            self.vt = np.zeros((num_params, 1))
            self.t = 0
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps

    def __call__(self, grads, learning_rate):
        if grads is not None:
            self.t += 1
            self.mt = self.beta1 * self.mt + (1 - self.beta1) * grads
            self.vt = self.beta2 * self.vt + (1 - self.beta2) * np.square(grads)
            m_hat = self.mt / (1 - self.beta1 ** self.t)
            v_hat = self.vt / (1 - self.beta2 ** self.t)
            delta = learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
            return delta
    
    def reset(self):
        if self.num_params != 0:
            self.mt = np.zeros(self.mt.shape)
            self.vt = np.zeros(self.vt.shape)
            self.t = 0