import torch
import torch.nn as nn
import numpy as np
import collections
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.parameter import Parameter
from utils.neural_networks import TCN, PositionalEncoding, RBFNetwork
import math

def const_shift(nu, t):
    try:
        t_init = t[0]
    except:
        t_init = t
    return t - 2*np.pi*((nu.phi*t_init)//(2*np.pi))/nu.phi

def modulo_shift(nu, t):
    return ((nu.phi*t) % (2*np.pi)) / nu.phi

def exp_2_renyi_div_normal_1d_fixed_sigma(mu_1, mu_2, sigma):
    """ Returns the exponential 2-Rényi divergence between two 
    1D normal distributions with same sigma.
    """
    return torch.exp((mu_1-mu_2)**2/(sigma**2))

def exp_2_renyi_div_normal_1d_diag_sigma(mu_P, sigma_P, mu_Q, sigma_Q,):
    """ Returns the exponential 2-Rényi divergence between two 
    1D normal distributions.
    """
    mu_P = mu_P.reshape(-1,1)
    mu_Q = mu_Q.reshape(1,-1)
    sigma_alpha = 2*sigma_Q - sigma_P
    
    # Clamping values before the exp to avoid inf
    temp = torch.clamp((mu_P-mu_Q)**2/(sigma_alpha**2), min=0, max=88)
    temp = torch.exp(temp)
    return temp*sigma_Q**2/(sigma_alpha*sigma_P)

def inv_exp_2_renyi_div_normal_1d_diag_sigma(mu_P, sigma_P, mu_Q, sigma_Q,):
    """ Returns the inverse of the exponential 2-Rényi divergence between two 
    1D normal distributions.
    """
    mu_P = mu_P.reshape(-1,1)
    mu_Q = mu_Q.reshape(1,-1)
    sigma_alpha = 2*sigma_Q - sigma_P
    
    # Clamping values before the exp to avoid inf
    temp = torch.clamp((mu_P-mu_Q)**2/(sigma_alpha**2), max=88, min=-40)
    temp = torch.exp(-temp)
    return temp*(sigma_alpha*sigma_P)/sigma_Q**2

def exp_2_renyi_div_normal_diag_sigma(mu_P, sigma_P, mu_Q, sigma_Q,):
    """ Returns the exponential 2-Rényi divergence between two 
    normal distributions.
    """
    mu_P = mu_P.unsqueeze(1)
    mu_Q = mu_Q.unsqueeze(0)
    sigma_Q = torch.diagonal(sigma_Q)
    sigma_P = torch.diagonal(sigma_P)
    sigma_alpha = 2* sigma_Q - sigma_P
    
    # Clamping values before the exp to avoid inf
    temp = torch.clamp(torch.prod((mu_P-mu_Q)**2/(sigma_alpha**2),axis=2), max=100)
    temp = torch.exp(temp)
    if any(np.isnan(temp.detach().numpy().reshape(-1))):
        print('NaN in Renyi divergence')
    return temp*torch.prod(sigma_Q**2)/torch.sqrt((torch.prod(sigma_alpha**2)*torch.prod(sigma_P**2)))

def inv_exp_2_renyi_div_normal_diag_sigma(mu_P, sigma_P, mu_Q, sigma_Q,):
    """ Returns the inverse of the exponential 2-Rényi divergence between two 
    normal distributions.
    """
    mu_P = mu_P.unsqueeze(1)
    mu_Q = mu_Q.unsqueeze(0)
    sigma_Q = torch.diagonal(sigma_Q)
    sigma_P = torch.diagonal(sigma_P)
    sigma_alpha = 2* sigma_Q - sigma_P
    
    # Clamping values before the exp to avoid inf
    temp = torch.clamp(torch.prod((mu_P-mu_Q)**2/(sigma_alpha**2),axis=2), max=800, min=-40)
    temp = torch.exp(-temp)
    if any(np.isnan(temp.detach().numpy().reshape(-1))):
        print('NaN in Renyi divergence')
    return temp*torch.sqrt((torch.prod(sigma_alpha**2)*torch.prod(sigma_P**2)))/torch.prod(sigma_Q**2)



def diag_multivariate_normal(x, mu, sigma):
    """ pdf of the multivariate normal distribution with diagonal
    covariance.
    """
    temp = -1/2*((x-mu)/sigma)**2
    temp = torch.exp(temp.sum(-1))
    return temp/(torch.sqrt(torch.tensor((2*np.pi)**len(sigma)))*torch.prod(sigma))

def log_diag_multivariate_normal(x, mu, sigma, log_sigma):
    """ log pdf of the multivariate normal distribution with diagonal
    covariance.
    """
    temp = ((x-mu)/sigma)**2
    temp = -1/2*(len(sigma)*math.log(2*math.pi) + temp.sum(-1))
    return  - torch.sum(log_sigma) + temp

def opt_phi(renyi_mat, psi, prior_zeta):
    """ Optimization for the value of phi.
    """
    temp = prior_zeta * psi / renyi_mat
    return temp / (torch.sum(psi / renyi_mat, axis=1)).reshape(-1,1)

def opt_psi(renyi_div_sqrt, phi, prior_mu):
    """ Optimization for the value of psi.
    """
    temp = prior_mu * phi * renyi_div_sqrt
    return temp / (torch.sum(phi * renyi_div_sqrt, axis=0))

def opt_variational_bound(renyi_mat, prior_mu, prior_zeta, n_optim=10, phi_init=None, 
            psi_init=None,):
    """ Convex optimization of the values of phi and psi.
    """
    prior_mu = prior_mu.reshape(1,-1)
    prior_zeta = prior_zeta.reshape(-1,1)
    phi = torch.matmul(prior_zeta, prior_mu) if phi_init is None else phi_init
    psi = torch.matmul(prior_zeta, prior_mu) if psi_init is None else psi_init
    renyi_div_sqrt = torch.sqrt(renyi_mat)
    for _ in range(n_optim):
        psi = opt_psi(renyi_div_sqrt, phi, prior_mu)
        phi = opt_phi(renyi_mat, psi, prior_zeta)
    return phi, psi


def get_discount_period(factor, period, flip=False):
    """ Returns the discount scheme for the consider period
    with the given discounting factor.
    """
    discount = torch.full((1,period), factor, dtype=float)
    discount = torch.cumprod(discount, dim=1)
    if flip:
        discount = torch.flip(discount, [1])
    return discount/factor




class HyperPolicy(nn.Module):
    def __init__(self, omega=None, gamma=None, alpha=None, beta=None,
                diag_sigma=True, var_bound=None, n_optim_var_bound=10,):
        super(HyperPolicy, self).__init__()
        self.param_optimized = []
        self.n_params = len(self.param_optimized)
        self.current_time = torch.zeros(1)

        # Set constants for surrogate objective computation
        if not None in [omega, gamma, alpha, beta]:
            self.omega = omega
            if omega == 1:
                self.sum_omega_2 = alpha
                self.C_omega = 1/alpha
            else:
                self.sum_omega_2 = (1-omega**(2*alpha))/(1-omega**2)
                self.C_omega = (1-omega)/(1-omega**(alpha))
            if gamma == 1:
                self.sum_gamma = beta
            else:
                self.sum_gamma = (1-gamma**(beta))/(1-gamma)
            self.alpha = alpha
            if gamma == omega:
                self.C_gamma_omega = alpha * gamma**(2*alpha)
            else:
                self.C_gamma_omega = (omega**(2*alpha)-gamma**(2*alpha))/(omega**2-gamma**2)
            
            # Set discount factors
            self.future_discount = get_discount_period(gamma, period=beta)
            self.future_discount_squared = self.future_discount**2 * gamma**(2*alpha)
            self.past_discount = get_discount_period(gamma, period=alpha)
            self.omega_discount = get_discount_period(omega, period=alpha,flip=True)
            
            # Initialize psi and phi for convex optimization 
            self.prior_zeta =  self.future_discount/self.sum_gamma
            self.prior_zeta = self.prior_zeta.float().T
            self.prior_mu = torch.full((1,alpha),1/alpha)
            self.n_optim_var_bound = n_optim_var_bound
            self.phi_var_bound = torch.matmul(self.prior_zeta, self.prior_mu) 
            self.psi_var_bound = torch.matmul(self.prior_zeta, self.prior_mu)

        # Set renyi divergence to 1D or more
        if self.n_policy_params==1:
            self.exp_renyi_div = exp_2_renyi_div_normal_1d_diag_sigma
            self.inv_exp_renyi_div = inv_exp_2_renyi_div_normal_1d_diag_sigma 
        else: 
            self.exp_renyi_div = exp_2_renyi_div_normal_diag_sigma
            self.inv_exp_renyi_div = inv_exp_2_renyi_div_normal_diag_sigma 
        
        # Set the functions related to the Gaussian distribution of the policy parameters
        if diag_sigma:
            self.log_nu = self.log_nu_diag
            self.theta_pdf = self.theta_pdf_diag
        else:
            self.log_nu = self.log_nu_non_diag
            self.theta_pdf = self.theta_pdf_non_diag

        
        
        # Choose variance variational bound
        if var_bound is None:
            var_bound = 'two_step_psi_first'
        if beta==1:
            self.variance_estimate = self.var_est_beta_1
            self.variance_loss_parallel = self.var_loss_beta_1
        elif var_bound=='two_step_psi_first':
            self.variance_estimate = self.var_est_two_step_psi_first 
            self.variance_loss_parallel = self.var_loss_two_step_psi_first
        elif var_bound=='two_step_phi_first':
            self.variance_estimate = self.var_est_two_step_phi_first  
            self.variance_loss_parallel = self.var_loss_two_step_phi_first
        elif var_bound=='uniform_psi':
            self.variance_estimate = self.var_est_uniform_psi 
            self.variance_loss_parallel = self.var_loss_uniform_psi
        elif var_bound=='uniform_phi':
            self.variance_estimate = self.var_est_uniform_phi 
            self.variance_loss_parallel = self.var_loss_uniform_phi
        elif var_bound=='cvx_optim_reset' or var_bound=='cvx_optim':
            self.keep_params_optim = False
            self.variance_estimate = self.var_est_cvx_optim
            self.variance_loss_parallel = self.var_loss_cvx_optim
        else: 
            raise NotImplementedError

    def get_grad(self):
        """ Returns the gradient for each hyper-parameter.
        """
        return torch.cat([p.grad.data.view(-1) for p in self.parameters() if p.grad is not None])

    def get_params(self):
        """ Returns the value of each hyper-parameter.
        """
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def get_n_param(self):
        """ Returns the number of hyper-parameter.
        """
        nb_param = 0
        for parameter in self.parameters():
            nb_param += parameter.numel()
        return nb_param

    def get_name_params(self):
        """ Returns the name of hyper-parameter.
        """
        return [p[0] for p in self.named_parameters()]
    
    def get_size_params(self):
        """ Returns the degree of each hyper-parameter.
        """
        return [p.numel() for p in self.parameters()]

    def set_n_params(self):
        """ Sets the value of attribute n_params.
        """
        self.n_params = len(self.get_params())

    def set_current_time(self, t):
        """ Sets the value of attribute current_time.
        """
        self.current_time = torch.tensor(t, dtype=torch.float).reshape(-1,1)
    
    def get_covariance(self):
        """ Returns the covariance.
        """
        return torch.exp(self.log_sigma_theta)*torch.eye(self.n_policy_params, dtype=torch.double)

    def theta_pdf_non_diag(self, theta, t):
        """ Returns the pdf for a Gaussian with non-diagonal covariance matrix.
        """
        covariance = self.get_covariance()
        dist = MultivariateNormal(loc=self.theta_mean(t), covariance_matrix=covariance)
        return torch.exp(dist.log_prob(theta))   

    def theta_pdf_diag(self, theta, t):
        """ Returns the pdf for Gaussian with diagonal covariance matrix.
        """
        return  diag_multivariate_normal(x=theta, 
                mu=self.theta_mean(t), sigma=torch.exp(self.log_sigma_theta))

    def log_nu_non_diag(self, thetas, t):
        """ Returns the log probability of a Gaussian with non-diagonal covariance matrix.
        """
        covariance = self.get_covariance()
        dist = MultivariateNormal(loc=self.theta_mean(t), covariance_matrix=covariance)
        return dist.log_prob(thetas)

    def log_nu_diag(self, thetas, t):
        """ Returns the log probability of a Gaussian with diagonal covariance matrix.
        """
        return log_diag_multivariate_normal(x=thetas, 
                mu=self.theta_mean(t), sigma=torch.exp(self.log_sigma_theta), 
                log_sigma=self.log_sigma_theta)

    def sample_policy(self, t, n_policies=1):
        """ Samples parameters of a policy a time t.
        """
        with torch.no_grad():
            theta_mean = self.theta_mean(t)
            theta_mean = theta_mean.repeat_interleave(n_policies, dim=0)

            if self.stochastic:
                theta = torch.normal(theta_mean, 
                        std=torch.exp(self.log_sigma_theta).unsqueeze(0).repeat(theta_mean.shape[0],1))
            else:
                theta = theta_mean
        
        # If more than one policy is to be sampled for each time.
        if n_policies==1 and isinstance(t, int):
            return self.policy(theta, not(self.stochastic))
        else:
            return [self.policy(thet, not(self.stochastic)) for thet in theta], \
                    theta.numpy().reshape(len(t), n_policies, -1).transpose(1,0,2)
    

    #### alpha-steps behind return ####
    def behind_return_loss(self, rewards, thetas, t):
        """ Compute the loss associated to the alpha-steps
        behind estimated return.
        """
        with torch.enable_grad():
            log_nu = self.log_nu(thetas, t)
            loss = -(rewards * log_nu * self.omega_discount * self.past_discount).sum()
        return loss
    
    #### beta-steps ahead return ####
    def future_return_estimate(self, rewards, thetas, t, alpha=60, beta=1):
        """ Compute the estimate of the beta-steps ahead estimated return.
        """
        assert thetas.shape[1] == rewards.shape[1], "Thetas and rewards have different lengths"
        assert thetas.shape[1] >= alpha, "Not enough data for the estimator"
        timesteps_alpha = torch.arange(t-alpha+1, t+1)
        timesteps_beta = torch.arange(t+1, t+beta+1)

        IS_est = []
        MIS_beta = self.theta_pdf(thetas.unsqueeze(2).repeat(1,1,alpha,1), timesteps_alpha).sum(axis=2)
        IS_weights = self.theta_pdf(thetas.unsqueeze(2).repeat(1,1,beta,1), timesteps_beta)#.transpose(1,2)
        IS_weights = (IS_weights * self.future_discount).sum(axis=2)
        IS_weights = IS_weights/MIS_beta
        IS_est  = IS_weights * rewards * self.omega_discount
        return torch.sum(IS_est), IS_est, IS_weights, MIS_beta

    def future_return_loss_parallel(self, rewards, thetas, t, alpha=60, beta=1):
        """ Compute the loss associated to the beta-steps
        ahead estimated return in parallel.
        """
        timesteps_alpha = torch.arange(t-alpha+1, t+1)
        
        with torch.enable_grad():
            _, IS_est, IS_weights, _ = self.future_return_estimate(rewards, thetas, t=t, alpha=alpha, beta=beta)
            
            past_log_nu = self.log_nu(thetas, timesteps_alpha)
            log_IS_weights = torch.log(torch.clamp(IS_weights, min=1e-40))
            loss = past_log_nu + log_IS_weights
            loss = IS_est.detach() * loss
            loss = -loss.sum()
        return loss, IS_weights.detach()
    
    #### variance regularization ####
    def var_loss_beta_1(self, t, alpha, beta,):
        """ Compute the loss associated to the variance for beta=1.
        """
        with torch.enable_grad():
            loss = -self.var_bound_beta_1(t, alpha, beta)
        return loss

    def var_bound_beta_1(self, t, alpha, beta,):
        """ Compute the upper-bound associated to the variance for beta=1.
        """
        covariance = self.get_covariance()
        timesteps_alpha = torch.arange(t-alpha+1, t+1)
        timesteps_beta = torch.arange(t+1, t+beta+1)
        renyi_div = self.exp_renyi_div(mu_P=self.theta_mean(timesteps_beta), sigma_P=covariance, 
                                                         mu_Q=self.theta_mean(timesteps_alpha), sigma_Q=covariance)
        return torch.sqrt(1+1/torch.sum(1/renyi_div))

    
    def var_loss_two_step_psi_first(self, t, alpha, beta,):
        """ Compute the loss associated to the variance for the 
        two_step_psi_first case.
        """
        with torch.enable_grad():
            loss = self.var_bound_two_step_psi_first(t, alpha, beta,)
        return loss

    def var_loss_two_step_phi_first(self, t, alpha, beta,):
        """ Compute the loss associated to the variance for the 
        two_step_phi_first case.
        """
        with torch.enable_grad():
            loss = self.var_bound_two_step_phi_first(t, alpha, beta,)
        return loss

    def var_loss_uniform_psi(self, t, alpha, beta,):
        """ Compute the loss associated to the variance for the 
        uniform_psi case.
        """
        with torch.enable_grad():
            loss = self.var_bound_uniform_psi(t, alpha, beta,)
        return loss

    def var_loss_uniform_phi(self, t, alpha, beta,):
        """ Compute the loss associated to the variance for the 
        uniform_phi case.
        """
        with torch.enable_grad():
            loss = self.var_bound_uniform_phi(t, alpha, beta,)
        return loss

    def var_loss_cvx_optim(self, t, alpha, beta,):
        """ Compute the loss associated to the variance for the 
        direct convex optimization case.
        """
        covariance = self.get_covariance()
        timesteps_alpha = torch.arange(t-alpha+1, t+1)
        timesteps_beta = torch.arange(t+1, t+beta+1)
        renyi_div = self.exp_renyi_div(mu_P=self.theta_mean(timesteps_beta), sigma_P=covariance, 
                mu_Q=self.theta_mean(timesteps_alpha), sigma_Q=covariance)
        with torch.no_grad():
            if self.keep_params_optim:
                # Using previous parameters as starting points for the new optimization.
                self.phi_var_bound, self.psi_var_bound = opt_variational_bound(renyi_div,
                        prior_mu=self.prior_mu, 
                        prior_zeta=self.prior_zeta, n_optim=self.n_optim_var_bound, 
                        phi_init=self.phi_var_bound, psi_init=self.psi_var_bound,)
                self.psi_var_bound += 1e-20
            else:
                # Starting the optimization with uniform guess.
                self.phi_var_bound, self.psi_var_bound = opt_variational_bound(renyi_div,
                        prior_mu=self.prior_mu, 
                        prior_zeta=self.prior_zeta, n_optim=self.n_optim_var_bound, )
                self.psi_var_bound += 1e-20
        
        self.zero_grad()
        with torch.enable_grad():
            loss = self.var_est_cvx_optim(t, alpha, beta,)
        return loss

    def var_bound_cvx_optim(self, t, alpha, beta,):
        """ Compute the loss associated to the variance for the 
        direct convex optimization case.
        """
        covariance = self.get_covariance()
        timesteps_alpha = torch.arange(t-alpha+1, t+1)
        timesteps_beta = torch.arange(t+1, t+beta+1)
        renyi_div = self.exp_renyi_div(mu_P=self.theta_mean(timesteps_beta), sigma_P=covariance, 
                mu_Q=self.theta_mean(timesteps_alpha), sigma_Q=covariance)
        renyi_mixture = torch.sum(self.phi_var_bound**2/self.psi_var_bound * renyi_div)
        return torch.sqrt(self.C_gamma_omega + (self.sum_gamma**2/alpha)*self.sum_omega_2*renyi_mixture)


    def var_bound_two_step_psi_first(self, t, alpha, beta,):
        """ Compute the loss associated to the variance for the 
        two_step_psi_first case.
        """
        covariance = self.get_covariance()
        timesteps_alpha = torch.arange(t-alpha+1, t+1)
        timesteps_beta = torch.arange(t+1, t+beta+1)
        inv_renyi_div = self.inv_exp_renyi_div(mu_P=self.theta_mean(timesteps_beta), sigma_P=covariance, 
                mu_Q=self.theta_mean(timesteps_alpha), sigma_Q=covariance).sum(axis=1)
        return torch.sqrt(
                            self.C_gamma_omega + 
                            self.sum_omega_2 * torch.sum(self.future_discount/torch.sqrt(inv_renyi_div))**2 
                        )

    def var_bound_two_step_phi_first(self, t, alpha, beta,):
        """ Compute the loss associated to the variance for the 
        two_step_phi_first case.
        """
        covariance = self.get_covariance()
        timesteps_alpha = torch.arange(t-alpha+1, t+1)
        timesteps_beta = torch.arange(t+1, t+beta+1)
        renyi_div = self.exp_renyi_div(mu_P=self.theta_mean(timesteps_beta), sigma_P=covariance, 
                mu_Q=self.theta_mean(timesteps_alpha), sigma_Q=covariance)
        temp = 1/torch.sum(self.future_discount.T*torch.sqrt(renyi_div), axis=0)
        return torch.sqrt(
                            self.C_gamma_omega + 
                            self.sum_omega_2 / torch.sum(temp**2)
                        )

    def var_bound_uniform_phi(self, t, alpha, beta,):
        """ Compute the loss associated to the variance for the 
        uniform_phi case.
        """
        covariance = self.get_covariance()
        timesteps_alpha = torch.arange(t-alpha+1, t+1)
        timesteps_beta = torch.arange(t+1, t+beta+1)
        renyi_div = self.exp_renyi_div(mu_P=self.theta_mean(timesteps_beta), sigma_P=covariance, 
                mu_Q=self.theta_mean(timesteps_alpha), sigma_Q=covariance)
        temp = torch.sum(self.future_discount.T*torch.sqrt(renyi_div), axis=0)
        return torch.sqrt(
                            self.C_gamma_omega + 
                            self.sum_omega_2 /(alpha**2) * torch.sum(temp**2)
                        )

    def var_bound_uniform_psi(self, t, alpha, beta,):
        """ Compute the loss associated to the variance for the 
        uniform_psi case.
        """
        covariance = self.get_covariance()
        timesteps_alpha = torch.arange(t-alpha+1, t+1)
        timesteps_beta = torch.arange(t+1, t+beta+1)
        inv_renyi_div = self.inv_exp_renyi_div(mu_P=self.theta_mean(timesteps_beta), sigma_P=covariance, 
                mu_Q=self.theta_mean(timesteps_alpha), sigma_Q=covariance).sum(axis=1)
        return torch.sqrt(
                            self.C_gamma_omega + 
                            beta * self.sum_omega_2 * torch.sum(self.future_discount_squared/inv_renyi_div)
                        )
 
    
###########################################################################
    
        
class NeuralHPolicy(HyperPolicy):
    """ Hyper-policy with positional encoding of time followed by
    linear layers.
    """
    def __init__(self, policy, state_dim, omega=None, gamma=None, alpha=None, beta=None, var_bound=None,
                n_optim_var_bound=10,
                stochastic=True, sigma_theta=1, horizon=100, n_neurons=[16,16], 
                pos_encoding_dim=4, use_pos_encoding=False, 
                learn_sigma=True, bound_output=False, **kwargs):
        self.n_policy_params = policy.n_params(state_dim, **kwargs['policy_args'])
        super(NeuralHPolicy, self).__init__(omega=omega, gamma=gamma, alpha=alpha, beta=beta, var_bound=var_bound, n_optim_var_bound=n_optim_var_bound)
        
        # Set hyper-policy model
        if learn_sigma:
            self.log_sigma_theta = Parameter(torch.tensor(sigma_theta, dtype=torch.double).repeat(self.n_policy_params), requires_grad=True) 
        else:
            self.log_sigma_theta = torch.tensor(sigma_theta, dtype=torch.double).repeat(self.n_policy_params) 
        self.stochastic = stochastic
        self.policy = policy

        # Set hyper-policy learnable parameters
        self.horizon = torch.tensor(horizon, dtype=float)
        # Positional encoding
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.positional_encoding = PositionalEncoding(pos_encoding_dim,)
            encoding_dim = pos_encoding_dim
        else:
            encoding_dim = 1
        # Feed-forward network
        try:
            n_neurons = [encoding_dim] + [n for n in n_neurons] + [self.n_policy_params]
        except:
            n_neurons = [encoding_dim, n_neurons, self.n_policy_params]
        layers = []
        for i, (n_1,n_2) in enumerate(zip(n_neurons[:-1], n_neurons[1:])):
            layers.append(nn.Linear(n_1,n_2))
            layers.append(nn.ReLU())
        if bound_output:
            layers[-1] = nn.Tanh()
        else:
            layers.pop()
        self.net = nn.Sequential(*layers)

        super(NeuralHPolicy, self).set_n_params()


    def theta_mean(self, t):
        t = t.reshape(-1,1) if torch.is_tensor(t) else torch.tensor(t, dtype=torch.float).reshape(-1,1)
        if self.use_pos_encoding:
            t = self.positional_encoding(t)
        else:
            t = (t - self.current_time+self.horizon/2)/self.horizon
        x = self.net(t).squeeze()
        return x.reshape(-1, self.n_policy_params)
    

###########################################################################
    
    
class NeuralTCNHPolicy(HyperPolicy):
    """ Hyper-policy with positional encoding of time followed by
    temporal convolutions.
    """
    def __init__(self, policy, state_dim, omega=None, gamma=None, alpha=None, beta=None, var_bound=None,
                n_optim_var_bound = 10,
                stochastic=True, sigma_theta=1, horizon=100, pos_encoding_dim=4,
                learn_sigma=True, bound_output=False, channels=[4,4], kernel_size=3,
                dropout=0, use_pos_encoding=False, **kwargs):
        self.n_policy_params = policy.n_params(state_dim, **kwargs['policy_args'])
        super(NeuralTCNHPolicy, self).__init__(omega=omega, gamma=gamma, alpha=alpha, beta=beta, var_bound=var_bound, n_optim_var_bound=n_optim_var_bound)
        
        # Set hyper-policy model
        if learn_sigma:
            self.log_sigma_theta = Parameter(torch.tensor(sigma_theta, dtype=torch.double).repeat(self.n_policy_params), requires_grad=True) 
        else:
            self.log_sigma_theta = torch.tensor(sigma_theta, dtype=torch.double).repeat(self.n_policy_params) 
        self.stochastic = stochastic
        self.policy = policy
        
        # Set hyper-policy learnable parameters
        self.horizon = torch.tensor(horizon, dtype=float)
        self.max_backward_lookup  = 2**(len(channels)-1)*(kernel_size-1)
        # Positional encoding
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.positional_encoding = PositionalEncoding(pos_encoding_dim,)
            encoding_dim = pos_encoding_dim
        else:
            pos_encoding_dim = 1
        # Temporal convolution
        self.tcn = TCN(input_size=encoding_dim, output_size=self.n_policy_params, 
                num_channels=channels, kernel_size=kernel_size, dropout=dropout)
        # Bounding output
        if bound_output:
            self.last_activation = nn.Tanh()
        else:
            self.last_activation = nn.Identity()

        super(NeuralTCNHPolicy, self).set_n_params()

    def forward(self, x, channel_last=True):
        x = self.tcn(x, channel_last)
        return self.last_activation(x).reshape(-1, self.n_policy_params)

    def theta_mean(self, t):
        t = t.reshape(-1,1) if torch.is_tensor(t) else torch.tensor(t, dtype=torch.float).reshape(-1,1)
        t_lookup = (t[0]-torch.arange(self.max_backward_lookup,0,-1,dtype=torch.float)).reshape(-1,1)
        t = torch.cat((t_lookup,t))
        if self.use_pos_encoding:
            t = self.positional_encoding(t - self.current_time)
        else:
            t = (t - self.current_time+self.horizon/2)/self.horizon
        x = self.forward(t.unsqueeze(0))[self.max_backward_lookup:].squeeze()
        return x.reshape(-1, self.n_policy_params)
    

###########################################################################
    
class StatPolicy(HyperPolicy):
    """ Stationary hyper-policy.
    """
    def __init__(self, policy, state_dim, omega=None, gamma=None, alpha=None, beta=None, var_bound=None,
                n_optim_var_bound = 10,
                stochastic=True, theta_mean=0, sigma_theta=1, learn_sigma=True, **kwargs):
        self.n_policy_params = policy.n_params(state_dim, **kwargs['policy_args'])
        super(StatPolicy, self).__init__(omega=omega, gamma=gamma, alpha=alpha, beta=beta, var_bound=var_bound, n_optim_var_bound=n_optim_var_bound)
        self.theta_mean_param = Parameter(torch.tensor(theta_mean, dtype=torch.double).repeat(self.n_policy_params).reshape(1,-1), requires_grad=True) 

        # Set hyper-policy model
        if learn_sigma:
            self.log_sigma_theta = Parameter(torch.tensor(sigma_theta, dtype=torch.double).repeat(self.n_policy_params), requires_grad=True) 
        else:
            self.log_sigma_theta = torch.tensor(sigma_theta, dtype=torch.double).repeat(self.n_policy_params) 
        self.stochastic = stochastic
        self.policy = policy

        super(StatPolicy, self).set_n_params()

    def theta_mean(self, t):
        t = t.reshape(-1,1) if torch.is_tensor(t) else torch.tensor(t, dtype=torch.float).reshape(-1,1)
        return self.theta_mean_param.repeat(len(t), 1)
    

    
###########################################################################
    
    
class SinPolicy(HyperPolicy):
    """ Sinusoidal hyper-policy.
    """
    def __init__(self, policy, state_dim, omega=None, gamma=None, alpha=None, beta=None, var_bound=None,
                n_optim_var_bound = 10, stochastic=True, sigma_theta=1, A=1, B=0, phi=0.1, psi=0,  
                learn_sigma=True, **kwargs):
        
        self.n_policy_params = policy.n_params(state_dim, **kwargs['policy_args'])
        super(SinPolicy, self).__init__(omega=omega, gamma=gamma, alpha=alpha, beta=beta, var_bound=var_bound, n_optim_var_bound=n_optim_var_bound)

        # Set hyper-policy model
        if learn_sigma:
            self.log_sigma_theta = Parameter(torch.tensor(sigma_theta, dtype=torch.double).repeat(self.n_policy_params), requires_grad=True) 
        else:
            self.log_sigma_theta = torch.tensor(sigma_theta, dtype=torch.double).repeat(self.n_policy_params)  
        self.stochastic = stochastic
        self.policy = policy

        # Set hyper-policy learnable parameters
        self.A = Parameter(torch.full((1, self.n_policy_params), A, dtype=torch.double), requires_grad=True) 
        self.B = Parameter(torch.full((1, self.n_policy_params), B, dtype=torch.double), requires_grad=True)   
        self.phi = Parameter(torch.full((1, self.n_policy_params), phi, dtype=torch.double), requires_grad=True)    
        self.psi = Parameter(torch.full((1, self.n_policy_params), psi, dtype=torch.double), requires_grad=True)
        super(SinPolicy, self).set_n_params() 

    def theta_mean(self, t):
        t = torch.tensor([t]) if not torch.is_tensor(t) else t
        t = t.reshape(-1,1).double()
        temp = self.A * torch.sin( torch.matmul(t, self.phi) + self.psi) + self.B
        return temp.reshape(-1, self.n_policy_params)


###########################################################################


class SinDriftPolicy(HyperPolicy):
    """ Sinusoidal hyper-policy with drift.
    """
    def __init__(self, policy, state_dim, omega=None, gamma=None, alpha=None, beta=None, var_bound=None,
                n_optim_var_bound = 10, stochastic=True, sigma_theta=1, A=1, B=0, phi=0.1, psi=0, 
                A_drift=1, B_drift=0, learn_sigma=True, **kwargs):
        self.n_policy_params = policy.n_params(state_dim, **kwargs['policy_args'])
        super(SinDriftPolicy, self).__init__(omega=omega, gamma=gamma, alpha=alpha, beta=beta, var_bound=var_bound, n_optim_var_bound=n_optim_var_bound)
        
        # Set hyper-policy model
        if learn_sigma:
            self.log_sigma_theta = Parameter(torch.tensor(sigma_theta, dtype=torch.double).repeat(self.n_policy_params), requires_grad=True) 
        else:
            self.log_sigma_theta = torch.tensor(sigma_theta, dtype=torch.double).repeat(self.n_policy_params) 
        self.stochastic = stochastic
        self.policy = policy

        # Set hyper-policy learnable parameters
        self.A_drift = Parameter(torch.full((1, self.n_policy_params), A_drift, dtype=torch.double), requires_grad=True) 
        self.B_drift = Parameter(torch.full((1, self.n_policy_params), B_drift, dtype=torch.double), requires_grad=True)   
        self.A = Parameter(torch.full((1, self.n_policy_params), A, dtype=torch.double), requires_grad=True) 
        self.B = Parameter(torch.full((1, self.n_policy_params), B, dtype=torch.double), requires_grad=True)   
        self.phi = Parameter(torch.full((1, self.n_policy_params), phi, dtype=torch.double), requires_grad=True)    
        self.psi = Parameter(torch.full((1, self.n_policy_params), psi, dtype=torch.double), requires_grad=True)
        
        super(SinDriftPolicy, self).set_n_params()

    def theta_mean(self, t):
        t = torch.tensor([t]) if not torch.is_tensor(t) else t
        t = t.reshape(-1,1).double()
        temp = self.A * torch.sin( torch.matmul(t, self.phi) + self.psi) + self.B
        temp = temp + self.A_drift * t + self.B_drift
        return temp.reshape(-1, self.n_policy_params)

###########################################################################
    
    
    
class MultisinPolicy(HyperPolicy):
    """ Hyper-policy with a sum of sinusoidal transformations.
    """
    def __init__(self, policy, state_dim, omega=None, gamma=None, alpha=None, beta=None, var_bound=None,
                n_optim_var_bound = 10, stochastic=True, sigma_theta=1., n_sin=2, A=1., B=0., phi=0.1, psi=0.,  
                learn_sigma=True, **kwargs):
        self.n_policy_params = policy.n_params(state_dim, **kwargs['policy_args'])
        super(MultisinPolicy, self).__init__(omega=omega, gamma=gamma, alpha=alpha, beta=beta, var_bound=var_bound, n_optim_var_bound=n_optim_var_bound)
        
        # Set hyper-policy model
        if learn_sigma:
            self.log_sigma_theta = Parameter(torch.tensor(sigma_theta, dtype=torch.double).repeat(self.n_policy_params), requires_grad=True) 
        else:
            self.log_sigma_theta = torch.tensor(sigma_theta, dtype=torch.double).repeat(self.n_policy_params) 
        self.stochastic = stochastic
        self.policy = policy

        # Set hyper-policy learnable parameters
        self.n_sin = n_sin
        rescale = torch.tensor([2**(-i) for i in range(self.n_sin)])
        temp = torch.full((1, self.n_policy_params, self.n_sin), A, dtype=torch.double)*rescale
        self.A = Parameter(temp, requires_grad=True) 
        temp = torch.full((1, self.n_policy_params, self.n_sin), B, dtype=torch.double)*rescale
        self.B = Parameter(temp, requires_grad=True)   
        temp = torch.full((1, self.n_policy_params, self.n_sin), phi, dtype=torch.double)*rescale
        self.phi = Parameter(temp, requires_grad=True)    
        temp = torch.full((1, self.n_policy_params, self.n_sin), psi, dtype=torch.double)*rescale
        self.psi = Parameter(temp, requires_grad=True)

        super(MultisinPolicy, self).set_n_params()

    def theta_mean(self, t):
        t = torch.tensor([t]) if not torch.is_tensor(t) else t
        t = t.reshape(-1,1,1)
        temp = self.A * torch.sin( t*self.phi + self.psi) + self.B
        temp = temp.sum(axis=2)
        return temp.reshape(-1, self.n_policy_params)


###########################################################################


class DriftPolicy(HyperPolicy):
    """ Hyper-policy with a drift.
    """
    def __init__(self, policy, state_dim, omega=None, gamma=None, alpha=None, beta=None, var_bound=None,
                n_optim_var_bound = 10, stochastic=True, sigma_theta=1, A=1, B=0, learn_sigma=True, **kwargs):
        self.n_policy_params = policy.n_params(state_dim, **kwargs['policy_args'])
        super(DriftPolicy, self).__init__(omega=omega, gamma=gamma, alpha=alpha, beta=beta, var_bound=var_bound, n_optim_var_bound=n_optim_var_bound)

        # Set hyper-policy model
        if learn_sigma:
            self.log_sigma_theta = Parameter(torch.tensor(sigma_theta, dtype=torch.double).repeat(self.n_policy_params), requires_grad=True) 
        else:
            self.log_sigma_theta = torch.tensor(sigma_theta, dtype=torch.double).repeat(self.n_policy_params) 
        self.stochastic = stochastic
        self.policy = policy

        # Set hyper-policy learnable parameters
        self.A = Parameter(torch.full((1, self.n_policy_params), A, dtype=torch.double), requires_grad=True) 
        self.B = Parameter(torch.full((1, self.n_policy_params), B, dtype=torch.double), requires_grad=True)

        super(DriftPolicy, self).set_n_params()

    def theta_mean(self, t):
        t = torch.tensor([t]) if not torch.is_tensor(t) else t
        t = t.reshape(-1,1)
        temp = self.A * t + self.B
        return temp.reshape(-1, self.n_policy_params)


###########################################################################  
    

class RBFPolicy(HyperPolicy):
    """ Hyper-policy with radial basis functions.
    """
    def __init__(self, policy, state_dim,  omega=None, gamma=None, alpha=None, beta=None, var_bound=None,
                n_optim_var_bound = 10, stochastic=True, sigma_theta=1, rbf_nodes=15, sigma_rbf=6, 
                infer_centers=True, infer_stds=True, init_linear_scale=0.2, learn_sigma=True, **kwargs):
        self.n_policy_params = policy.n_params(state_dim, **kwargs['policy_args'])
        super(RBFPolicy, self).__init__(omega=omega, gamma=gamma, alpha=alpha, beta=beta, var_bound=var_bound, n_optim_var_bound=n_optim_var_bound)
        
        # Set hyper-policy model
        if learn_sigma:
            self.log_sigma_theta = Parameter(torch.tensor(sigma_theta, dtype=torch.double).repeat(self.n_policy_params), requires_grad=True) 
        else:
            self.log_sigma_theta = torch.tensor(sigma_theta, dtype=torch.double).repeat(self.n_policy_params) 
        self.stochastic = stochastic
        self.policy = policy

        # Set hyper-policy learnable parameters
        self.rbf_nodes = rbf_nodes
        self.infer_centers = infer_centers
        self.infer_stds = infer_stds
        centers = torch.linspace(0, alpha, rbf_nodes)
        self.RBF_net = RBFNetwork(centers, sigma_rbf, n_output=self.n_policy_params, 
                infer_centers=infer_centers, infer_stds=infer_stds, init_linear_scale=init_linear_scale)
        
        super(RBFPolicy, self).set_n_params()
        
    def theta_mean(self, t):
        return self.RBF_net(t).reshape(-1, self.n_policy_params)
    
    def set_current_time(self, t):
        return super().set_current_time(t)
    

###########################################################################
    

class NeuralSinPolicy(HyperPolicy):
    """ Sinusoidal hyper-policy followed by linear layers.
    """
    def __init__(self, policy, state_dim, omega=None, gamma=None, alpha=None, beta=None, var_bound=None,
                n_optim_var_bound = 10,
                stochastic=True, sigma_theta=1, A=1, B=0, phi=0.1, psi=0,
                horizon=100, n_neurons=[16,16], learn_sigma=True, **kwargs):
        self.n_policy_params = policy.n_params(state_dim, **kwargs['policy_args'])
        super(NeuralSinPolicy, self).__init__(omega=omega, gamma=gamma, alpha=alpha, beta=beta, var_bound=var_bound, n_optim_var_bound=n_optim_var_bound)
        
        # Set hyper-policy model
        if learn_sigma:
            self.log_sigma_theta = Parameter(torch.tensor(sigma_theta, dtype=torch.double).repeat(self.n_policy_params), requires_grad=True) 
        else:
            self.log_sigma_theta = torch.tensor(sigma_theta, dtype=torch.double).repeat(self.n_policy_params) 
        self.stochastic = stochastic
        self.policy = policy

        # Set hyper-policy learnable parameters
        self.horizon = torch.tensor(horizon)
        # Feed-forward network
        try:
            n_neurons = [self.n_policy_params*2] + [n for n in n_neurons] + [self.n_policy_params]
        except:
            n_neurons = [self.n_policy_params*2, n_neurons, self.n_policy_params]
        layers = []
        for i, (n_1,n_2) in enumerate(zip(n_neurons[:-1], n_neurons[1:])):
            layers.append(nn.Linear(n_1,n_2))
            if i==0: # get an input in [-1, 1] for the rest of the network, whatever t
                layers.append(nn.Tanh())
            elif i==len(n_neurons)-2:
                pass
            else:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        # Sinusoidal parameters
        self.A = Parameter(torch.full((1, self.n_policy_params), A, dtype=torch.double), requires_grad=True) 
        self.B = Parameter(torch.full((1, self.n_policy_params),B, dtype=torch.double), requires_grad=True)   
        self.phi = Parameter(torch.full((1, self.n_policy_params), phi, dtype=torch.double), requires_grad=True)    
        self.psi = Parameter(torch.full((1, self.n_policy_params), psi, dtype=torch.double), requires_grad=True)

        super(NeuralSinPolicy, self).set_n_params()

    def theta_mean(self, t):
        t = t.reshape(-1,1) if torch.is_tensor(t) else torch.tensor(t, dtype=torch.float).reshape(-1,1)
        temp = self.A * torch.sin( torch.matmul(t.double(), self.phi) + self.psi) + self.B
        t = (t - self.current_time)/self.horizon
        t = t.repeat(1, self.n_policy_params)
        temp = temp.float()
        x = self.net(torch.cat([t,temp],axis=1)).squeeze()
        return x
    

