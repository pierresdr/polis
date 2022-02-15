import numpy as np

class Process(object):
    """ Parent class process.
    """
    def __init__(self):
        pass

    def reset(self):
        pass


class SinProcess(Process):
    """ Sinusoidal process.
    """
    def __init__(self, A=1, B=0, phi=1, psi=0,):
        self.A = A
        self.B = B
        self.phi = phi
        self.psi = psi

    def evaluate(self, t):
        return self.A * np.sin(self.phi*t + self.psi) + self.B


class DriftProcess(Process):
    """ Process with drift.
    """
    def __init__(self, A=1, B=0):
        self.A = A
        self.B = B
        
    def evaluate(self, t):
        return np.array([self.A*t + self.B])


class DriftSinProcess(Process):
    """ Sinusoidal process with drift.
    """
    def __init__(self, A=1, B=0, phi=1, psi=0):
        self.A = A
        self.B = B
        self.phi = phi
        self.psi = psi
        
    def evaluate(self, t):
        return np.array([self.A*t + self.B + self.A*np.sin(self.phi*t + self.psi)])

    
class StepProcess(Process):
    """ Process with step.
    """
    def __init__(self, sigma_r=1, sample_every=1):
        self.history = []
        self.sigma_r = sigma_r 
        self.sample_every = sample_every

    def reset(self):
        self.history = []

    def evaluate(self, t):
        try:
            return self.history[t]
        except:
            new_steps = t - len(self.history)
            values_to_sample = 1 + new_steps//self.sample_every
            new_values = np.random.normal(scale=self.sigma_r, size=values_to_sample).repeat(self.sample_every)
            self.history = np.concatenate((self.history, new_values))
            return self.history[t]


class VasicekProcess(Process):
    """ Vasicek process.
    """
    def __init__(self, A_r=0.1, B_r=0, sigma_r=1, process_init=0,):
        self.init = process_init
        self.history = np.array([process_init])
        self.A_r = A_r
        self.B_r = B_r
        self.sigma_r = sigma_r 
    
    def reset(self):
        self.history = np.array([self.init])

    def evaluate(self, t):
        try:
            return self.history[t]
        except:
            new_steps = t - len(self.history)
            values_to_sample = 1 + new_steps
            new_values = np.zeros(values_to_sample)
            temp = self.history[-1]
            for i in range(values_to_sample):
                temp = temp + self.A_r*(self.B_r-temp) + np.random.normal(scale=self.sigma_r)
                new_values[i] = temp
            self.history = np.concatenate((self.history, new_values))
            return self.history[t]
        

class NullProcess(Process):
    """ Null process.
    """
    def __init__(self,):
        pass

    def evaluate(self, t):
        return 0
