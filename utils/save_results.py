
import json, os
import numpy as np
import torch

def save_run(nu=None, optimizer=None, state_epoch=None, reward_epoch=None, action_epoch=None, theta_epoch=None,
             variance_epoch=None, param_epoch=None, MIS_w_epoch=None, J2_epoch=None, loss_epoch=None, J1_epoch=None, 
             seed=0, save_folder='save'):
    
    if nu is not None:
        with open(os.path.join(save_folder,'hyperpolicy_classes_{}.txt'.format(seed)), 'w') as f:
            json.dump(nu.__class__.__name__,f)
            
    # Save numpy variables
    list_save = [{'variable': param_epoch, 'name': 'params_optim'},
                {'variable': state_epoch, 'name': 'states_optim'},
                {'variable': action_epoch, 'name': 'actions_optim'},
                {'variable': theta_epoch, 'name': 'thetas_optim'},
                {'variable': reward_epoch, 'name': 'rewards_optim'},
                {'variable': MIS_w_epoch, 'name': 'MIS_w_optim'},
                {'variable': J1_epoch, 'name': 'J1_optim'},
                {'variable': J2_epoch, 'name': 'J2_optim'},
                {'variable': variance_epoch, 'name': 'var_optim'},
                {'variable': loss_epoch, 'name': 'loss_optim'},]
    for d in list_save:
        save_numpy(**d, save_folder=save_folder, seed=seed)

    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(save_folder,'optimizer_{}.txt'.format(seed))) 


def save_lifelong(nu=None, optimizer=None, params_optim=None, states_optim=None, 
                actions_optim=None, thetas_optim=None, rewards_optim=None, MIS_w_optim=None, 
                J2_optim=None, var_optim=None, loss_optim=None, states=None,
                thetas=None, rewards=None, actions=None, ns_process=None, info_process=None, 
                seed=0, save_folder='save'):
    
    if nu is not None:
        with open(os.path.join(save_folder,'hyperpolicy_classes_{}.txt'.format(seed)), 'w') as f:
            json.dump(nu.__class__.__name__,f)

    # Save numpy variables
    list_save = [{'variable': params_optim, 'name': 'params_optim'},
                {'variable': states_optim, 'name': 'states_optim'},
                {'variable': actions_optim, 'name': 'actions_optim'},
                {'variable': thetas_optim, 'name': 'thetas_optim'},
                {'variable': rewards_optim, 'name': 'rewards_optim'},
                {'variable': MIS_w_optim, 'name': 'MIS_w_optim'},
                {'variable': J2_optim, 'name': 'J2_optim'},
                {'variable': var_optim, 'name': 'var_optim'},
                {'variable': loss_optim, 'name': 'loss_optim'},
                {'variable': rewards, 'name': 'rewards_play'},
                {'variable': thetas, 'name': 'thetas_play'},
                {'variable': states, 'name': 'states_play'},
                {'variable': ns_process, 'name': 'ns_process_play'},
                {'variable': info_process, 'name': 'info_process_play'},
                {'variable': actions, 'name': 'actions_play'},]
    
    for d in list_save:
        save_numpy(**d, save_folder=save_folder, seed=seed)

    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(save_folder,'optimizer_{}.txt'.format(seed))) 


def save_numpy(variable, name, save_folder, seed=0):
    if variable is not None:
        with open(os.path.join(save_folder,'{}_{}.txt'.format(name,seed)), 'wb') as f:
            np.save(f, variable)