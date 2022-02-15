import gym
import numpy as np
from gym import spaces
import scipy.stats as ss
import gym_bandits.processes as proc


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity(x):
    return x


class ContextualBandit(gym.Env):
    """ Contextual bandit parent class.
    """
    def __init__(self, t_init=0, mean_reward=False):
        self.t_init = t_init
        self.sample_reward = identity if mean_reward else ss.bernoulli.rvs
        self.gamma = 1.0
        self.info_process = ['reward mean']
        self.action_space = spaces.Discrete(1)

    def reset(self, t=None,):
        self.ns_history = []

        if t is None:
            self.t = self.t_init
        else:
            self.t = t

        # Reset stochastic processes
        self.c_process.reset()
        self.r_process.reset()

        # Sample context
        c, _ = self.sample_context()
        self.state = c
        self.ns_history.append(c)
        return self.state

    def get_reward(self, context, action, t):
        mean_r = self.r_process.evaluate(t)
        p_t = sigmoid( 50 * (context - mean_r) )  
        rewards = np.stack([self.sample_reward(1-p_t).squeeze(),self.sample_reward(p_t).squeeze(),]).reshape(2,-1)
        action = np.array(action).reshape(1,-1).astype(int)
        rewards = np.take_along_axis(rewards,action,axis=0)
        return rewards.squeeze(), mean_r

    def sample_context(self):
        mean_c = self.c_process.evaluate(self.t)
        return [ss.norm.rvs(loc=mean_c, scale=self.sigma_c)], mean_c

    def step(self, action):
        r, mean_r = self.get_reward(self.state, action, self.t)
        self.state, mean_c = self.sample_context()
        self.ns_history.append(self.state)
        self.t += 1
        return self.state, r, False, {'reward mean': mean_r}
    
    def replay_step(self, state, action, t):
        r, mean_r = self.get_reward(state, action, t)
        return self.ns_history[t], r, False, {'reward mean': mean_r}

    
class PeriodicBandit(ContextualBandit):
    """ Contextual bandit with sinusoidal context.
    """
    def __init__(self, sigma_c=1, A_c=1, A_r=1, B_c=0, B_r=0, phi_c=1, 
            phi_r=1,  psi_c=0, psi_r=0, t_init=0, mean_reward=False, **kwargs):
        super(PeriodicBandit, self).__init__(t_init=t_init, mean_reward=mean_reward)
        self.state_dim = 1
        self.ns_dim = 1
        self.sigma_c = sigma_c
        self.c_process = proc.SinProcess(A=A_c, B=B_c, phi=phi_c, psi=psi_c)
        self.r_process = proc.SinProcess(A=A_r, B=B_r, phi=phi_r, psi=psi_r)
    

class DriftBandit(ContextualBandit):
    """ Contextual bandit with drift context.
    """
    def __init__(self, sigma_c=1, A_c=1, A_r=1, B_c=0, B_r=0, t_init=0, mean_reward=False, **kwargs):
        super(DriftBandit, self).__init__(t_init=t_init, mean_reward=mean_reward)
        self.sigma_c = sigma_c
        self.state_dim = 1
        self.ns_dim = 1
        self.c_process = proc.DriftProcess(A=A_c, B=B_c,)
        self.r_process = proc.DriftProcess(A=A_r, B=B_r,)

class DriftSinBandit(ContextualBandit):
    """ Contextual bandit with sinusoidal context with drift.
    """
    def __init__(self, sigma_c=1, A_c=1, A_r=1, B_c=0, B_r=0, t_init=0, 
                 phi_c=1, psi_c=0,mean_reward=False, **kwargs):
        super(DriftSinBandit, self).__init__(t_init=t_init, mean_reward=mean_reward)
        self.sigma_c = sigma_c
        self.state_dim = 1
        self.ns_dim = 1
        self.c_process = proc.DriftSinProcess(A=A_c, B=B_c, phi=phi_c, psi=psi_c)
        self.r_process = proc.DriftProcess(A=A_r, B=B_r,)
    

class StepBandit(ContextualBandit):
    """ Contextual bandit with context with step.
    """
    def __init__(self, sigma_c=1, sigma_r=1, sample_every=1, t_init=0, mean_reward=False, **kwargs):
        super(StepBandit, self).__init__(t_init=t_init, mean_reward=mean_reward)
        self.sigma_c = sigma_c
        self.c_process = proc.NullProcess()
        self.r_process = proc.StepProcess(sigma_r=sigma_r, sample_every=sample_every,)


class VasicekBandit(ContextualBandit):
    """ Contextual bandit with Vasicek context.
    """
    def __init__(self, sigma_c=1, sigma_r=0.1, A_r=1, B_r=0, t_init=0, mean_reward=False, **kwargs):
        super(VasicekBandit, self).__init__(t_init=t_init, mean_reward=mean_reward)
        self.state_dim = 1
        self.ns_dim = 1
        self.sigma_c = sigma_c
        #self.r_process = proc.NullProcess()
        self.c_process = proc.VasicekProcess(A_r=A_r, B_r=B_r, sigma_r=sigma_r,)
        self.r_process = self.c_process