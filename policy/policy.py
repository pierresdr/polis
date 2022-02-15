import numpy as np
from utils.neural_networks import TCN, PositionalEncoding
import torch.nn as nn
import torch
import math

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig

class OnlyBiasPolicy(object):
    """ Constant policy.
    """
    @staticmethod
    def set_bounds(bound_low, bound_high):
        pass

    @staticmethod
    def n_params(state_dim, **policy_args):
        return 1 

    def __init__(self, theta, stochastic,):
        self.theta = theta.numpy().reshape(-1)
        self.theta_B = self.theta[0]
        self.stochastic = stochastic

    def get_param_names(self,):
        return ['theta_B']

    def sample_action(self, state):
        return self.theta_B


class LinearPolicy(object):
    """ Linear policy.
    """
    @staticmethod
    def set_bounds(bound_low, bound_high):
        pass

    @staticmethod
    def n_params(state_dim, **policy_args):
        return state_dim + 1 

    def __init__(self, theta, stochastic,):
        self.theta = theta.numpy().reshape(-1)
        self.theta_A = self.theta[:-1]
        self.theta_B = self.theta[-1]
        self.stochastic = stochastic

    def get_param_names(self,):
        return ['theta_A_{}'.format(i) for i in range(len(self.theta_A))] + ['theta_B']

    def sample_action(self, state):
        return np.matmul(self.theta_A, state) + self.theta_B

class LinearBoundedPolicy(object):
    """ Bounded linear policy.
    """
    @staticmethod
    def set_bounds(bound_low, bound_high):
        LinearBoundedPolicy.ACTION_LOW = bound_low
        LinearBoundedPolicy.ACTION_DIFF = bound_high - bound_low
    
    @staticmethod
    def n_params(state_dim, **policy_args):
        return state_dim + 1 

    def __init__(self, theta, stochastic,):
        self.theta = theta.numpy().reshape(-1)
        self.theta_A = self.theta[:-1]
        self.theta_B = self.theta[-1]
        self.stochastic = stochastic

    def get_param_names(self,):
        return ['theta_A_{}'.format(i) for i in range(len(self.theta_A))] + ['theta_B']

    def sample_action(self, state):
        # return np.tanh(np.matmul(self.theta_A, state) + self.theta_B)
        return (sigmoid(np.matmul(self.theta_A, state) + self.theta_B) * LinearBoundedPolicy.ACTION_DIFF \
                + LinearBoundedPolicy.ACTION_LOW).item()

class LinearNoBiasPolicy(object):
    """ Linear policy without bias.
    """
    @staticmethod
    def set_bounds(bound_low, bound_high):
        pass

    @staticmethod
    def n_params(state_dim, **policy_args):
        return state_dim

    def __init__(self, theta, stochastic,):
        self.theta = theta.numpy().reshape(-1)
        self.theta_A = self.theta
        self.stochastic = stochastic

    def get_param_names(self,):
        return ['theta_A_{}'.format(i) for i in range(len(self.theta_A))]

    def sample_action(self, state):
        return np.matmul(self.theta_A, state)


class CategoricalPolicy(object):
    """ Categorical policy.
    """
    @staticmethod
    def set_bounds(bound_low, bound_high):
        pass

    @staticmethod
    def n_params(state_dim, **policy_args):
        return state_dim
        
    def __init__(self, theta, stochastic,):
        self.theta = theta.numpy().reshape(-1)
        self.stochastic = stochastic
    
    def get_param_names(self,):
        return ['theta_{}'.format(i) for i in range(len(self.theta))]

    def sample_action(self, state):
        return (state - self.theta > 0).item()



class TCNPolicy(object):
    """  Policy with temporal convolution.
    """
    @staticmethod
    def set_bounds(bound_low, bound_high):
        LinearBoundedPolicy.ACTION_LOW = bound_low
        LinearBoundedPolicy.ACTION_DIFF = bound_high - bound_low

    @staticmethod
    def n_params(state_dim, memory_size=10, hidden_size=4, kernel_size=3, channels=[4,4], dropout=0, **policy_args):
        TCNPolicy.tcn = TCN(input_size=int(state_dim/memory_size), output_size=hidden_size, 
                num_channels=channels, kernel_size=kernel_size, dropout=dropout)
        TCNPolicy.linear = nn.Linear(hidden_size*memory_size,1)
        TCNPolicy.log_sigma = nn.Parameter(torch.tensor(-1, dtype=torch.double))
        TCNPolicy.real_state_dim = int(state_dim/memory_size)
        TCNPolicy.memory_size = memory_size
        nb_param = 0
        for parameter in TCNPolicy.tcn.parameters():
            nb_param += parameter.numel()
        for parameter in TCNPolicy.linear.parameters():
            nb_param += parameter.numel()
        return nb_param
        
    def __init__(self, theta, stochastic,):
        self.theta = theta.numpy().reshape(-1)
        self.stochastic = stochastic
        idx_param = 0
        for p in TCNPolicy.tcn.parameters():
            n_params = p.numel()
            p.data = torch.tensor(self.theta[idx_param:idx_param+n_params])\
                    .reshape(p.shape)
            idx_param += n_params
        for p in TCNPolicy.linear.parameters():
            n_params = p.numel()
            p.data = torch.tensor(self.theta[idx_param:idx_param+n_params])\
                    .reshape(p.shape)
            idx_param += n_params
    
    def get_param_names(self,):
        return ['theta_{}'.format(i) for i in range(len(self.theta))]

    def sample_action(self, state,):
        x = torch.tensor(state).unsqueeze(0)
        x = TCNPolicy.tcn(x, channel_last=True).reshape(-1)
        x = TCNPolicy.linear(x)
        x = (sigmoid(x.item()) * LinearBoundedPolicy.ACTION_DIFF \
                    + LinearBoundedPolicy.ACTION_LOW).item()
        if self.stochastic:
            x += np.random.random() * np.exp(TCNPolicy.log_sigma.item())
        return x

    def sample_action_parallel(self, state,):
        x = torch.tensor(state)
        x = x.reshape(-1, TCNPolicy.memory_size, TCNPolicy.real_state_dim)
        x = TCNPolicy.tcn(x, channel_last=True).reshape(x.size(0),-1)
        x = TCNPolicy.linear(x)
        x = torch.sigmoid(x) * LinearBoundedPolicy.ACTION_DIFF \
                    + LinearBoundedPolicy.ACTION_LOW
        if self.stochastic:
            x = x + torch.randn_like(x) * torch.exp(TCNPolicy.log_sigma)
        return x.detach().numpy().reshape(-1)


