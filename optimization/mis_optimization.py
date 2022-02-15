
import numpy as np
import scipy, tqdm
from utils.sampling import replay_parallel
import torch
import time


def get_time(current_time, name=''):
    print('{}: \t {:.2f} seconds'.format(name,time.time()-current_time))
    return time.time()


def optimize(nu, env, optimizer, t, state_init, epochs_optim, alpha, 
            beta=1, lamb_J_behind=1, lamb_J_ahead=0, lamb_v=0, 
            R_inf=1, grad_replicas=1, seed=None, show_tqdm=True,):

    torch.autograd.set_detect_anomaly(True)
    
    # Initializa statistics
    statistics = {
        'params': np.zeros((epochs_optim+1, nu.n_params)),
        'grads': np.zeros((epochs_optim, nu.n_params)),
        'states': np.zeros((epochs_optim, alpha+1, env.state_dim)),
        'rewards': np.zeros((epochs_optim, alpha)),
        'actions': np.zeros((epochs_optim, alpha)),
        'thetas': np.zeros((epochs_optim, alpha, nu.n_policy_params)),
        'J2': np.zeros(epochs_optim),
        'variance': np.zeros(epochs_optim),
        'MIS_weights': np.zeros((epochs_optim, beta, alpha)),
        'losses': np.zeros((epochs_optim, 1)),
        'infos': {k:np.zeros((epochs_optim, alpha)) for k in env.info_process},
    }
    statistics['params'][0] = nu.get_params()
    nu.set_current_time(t)

    # Optimization
    if show_tqdm:
        iterator = tqdm.tqdm(range(epochs_optim))
    else:
        iterator = range(epochs_optim)
    for i in iterator:
        loss = torch.zeros(grad_replicas)
        optimizer.zero_grad()

        # Collect grad_replicas samples
        states, rewards, thetas, actions, info_processes =\
                replay_parallel(nu, env, t, state_init=state_init, alpha=alpha, n_play=grad_replicas)
        
        # Compute the loss
        loss, MIS_w = objective_loss(lamb_J_behind, lamb_J_ahead, lamb_v, rewards, thetas, nu, t, 
                        R_inf=R_inf, alpha=alpha, beta=beta,)
        if any(np.isnan(loss.detach().numpy())):
            print('Nan loss')
        loss.backward()
        optimizer.step()

        if any(np.isnan(nu.get_params().numpy())):
            print('Nan parameter')
        
        # Save variables
        statistics['params'][i+1] = nu.get_params()
        statistics['losses'][i] = loss.item()
        statistics['rewards'][i] = rewards[-1]
        statistics['states'][i] = states[-1]
        statistics['grads'][i] = nu.get_grad()
        if MIS_w is not None:
            statistics['MIS_weights'][i] = MIS_w[-1]
        statistics['actions'][i] = actions[-1]
        statistics['thetas'][i] = thetas[-1]
        for k,v in info_processes.items():
            statistics['infos'][k][i] = v[-1]
        with torch.no_grad():
            statistics['J_ahead'][i], _, _ ,_ = nu.future_return_estimate(torch.tensor(rewards), torch.tensor(thetas), t, alpha=alpha, beta=beta)
        
    return nu, optimizer, statistics


def objective_loss(lamb_J_behind, lamb_J_ahead, lamb_v, rewards, thetas, nu, t,
            alpha=60, beta=1,):
    assert rewards.shape[1]== alpha, 'Inconsistent length of rewards ({}) for alpha {}'.format(len(rewards),alpha)
    assert thetas.shape[1]== alpha, 'Inconsistent length of rewards ({}) for alpha {}'.format(len(thetas),alpha)
    rewards = torch.from_numpy(rewards)
    thetas = torch.from_numpy(thetas)
    loss = torch.zeros(1)
    MIS_weights = None

    # alpha-steps behind return loss
    if lamb_J_behind != 0:
        temp =  nu.behind_return_loss(rewards, thetas, torch.arange(t-alpha+1,t+1))
        loss = loss + lamb_J_behind * temp

    # beta-steps ahead return loss
    if lamb_J_ahead != 0:
        temp, MIS_weights = nu.ahead_return_loss_parallel(rewards, thetas, t, alpha=alpha, beta=beta)
        loss = loss + lamb_J_ahead * temp

    # variance regularization loss
    if lamb_v != 0:
        temp = nu.variance_loss_parallel(t, alpha=alpha, beta=beta,)*rewards.shape[0] # take the loss for each grad_replicas
        loss = loss + lamb_v * temp
    return loss, MIS_weights
