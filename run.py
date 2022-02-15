

import fire, os, json, tqdm, copy, datetime, logging
import numpy as np
import torch, gym, gym_bandits
from typing import Type, Union
import torch.optim as optim

from utils.plots import plots
from optimization.mis_optimization import optimize
import policy.hyperpolicy as hypol
import policy.policy as pol
from utils.sampling import sampling, resume_sampling
from utils.save_results import save_lifelong


# Logging level
logging.basicConfig(level='INFO', format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
        
        
def run_lifelong(n_behave_samples: int, n_target_samples: int, epochs_optim: int, 
            optim_every: int, hyperpolicy_class: Type(hypol.HyperPolicy), 
            optimizer_class=Type(optim.RMSprop),
            policy_class=Type(pol.LinearPolicy), env: str="Dam-v0", env_args: dict={}, 
            save_folder: str='save',
            alpha: int=60, beta: int=1, learning_rate: float=1e-3, lamb_J_ahead: float=1, 
            lamb_J_behind: float=1, lamb_v: float=1e-5, 
            hyperpolicy_args: dict={}, policy_args: dict={},
            seeds: Union[int,list]=0, 
            param_based: bool=True, 
            omega: float=1, 
            gamma: float=1, 
            var_bound: str='two_step_psi_first',
            n_optim_var_bound: int=10, 
            grad_samples: int=1,
            grad_steps: int=1, 
            
            mean_theta_init: float=1,
            sigma_theta_behavioural: float=1, 
            sigma_theta_init: float=1, 
            learn_sigma: bool=True, 

            save_extra=False,
            use_modulo=False,
            ) -> None:
    """

    This function is used to train POLIS on a given environment in a lifelong 
    learning setting.

    Args:
        n_behave_samples (int): Number of samples for the behavioural period.
        n_target_samples (int): Number of samples for the target period.
        epochs_optim (int): The second parameter.
        hyperpolicy_class (HyperPolicy): Class of the hyper-policy.
        hyperpolicy_args (dict): Extra parameters for the hyper-policy.
        policy_args (dict): Extra parameters for the policy.
        policy_class (Poliy): Class of the policy.
        optimizer_class (): Class of the optimizer in Pytorch.
        grad_steps (int): Number of gradient step per training. 
        grad_samples (int): Number of samples to estimate the gradient.
        var_bound (str): Type of variational bound for Rényi divergence involved
                in the bound on the variance.
        n_optim_var_bound (int): Number of optimization steps for the variational
                bound when set to convex optimization.
        env (int): Name of the gym environment.
        env_args (dict): Parameters of the environment.
        omega (float): Time-based discount parameter for the estimators. 
        gamma (float): Task's discounting factor. 
        save_folder (str): Folder where plots, models and statistics are saved.
        alpha (int): Number of steps to consider for the alpha-steps behind expected return.
        beta (int): Number of steps to consider for the beta-steps ahead expected return.
        lamb_J_behind (float): Weight for the alpha-steps behind expected return in the loss, 
                set to 1 for the framework of the paper.
        lamb_J_ahead (float): Weight for the beta-steps ahead expected return in the loss, 
                set to 1 for the framework of the paper. 
        lamb_v (float): Weight for the variance regularization of the loss. 
        seeds (Union[int,list]): Seeds to evaluate.
        param_based (bool): Parameter (true) or action (false) based.
        mean_theta_init (float): Initial value of the mean for the Gaussian
                distribution sampling theta.
        sigma_theta_behavioural (float): Value of sigma for the Gaussian
                distribution sampling theta during behavioural period.
        sigma_theta_init (float): Initial value of sigma for the Gaussian
                distribution sampling theta.
        learn_sigma (bool): Whether or not to allow the gradient to flow to
                sigma.
                
    Returns:

    """

    # Save parameters of the current run inside json file
    params = copy.deepcopy(locals())
    for k,v in params.items():
        if isinstance(v,type): 
            params[k] = v.__name__

    # Create folder 
    head, tail = os.path.split(save_folder)
    tail = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_") + tail
    save_folder = os.path.join(head, tail)
    try:
        os.makedirs(save_folder)
    except:
        pass
    with open(os.path.join(save_folder,'parameters.json'), 'w') as f:
        json.dump(params, f)

    
    seeds = seeds if hasattr(seeds,'__len__') else [seeds]
    hyperpolicy_class = eval(hyperpolicy_class) if isinstance(hyperpolicy_class, str) else hyperpolicy_class
    policy_class = eval(policy_class) if isinstance(policy_class, str) else policy_class
    optimizer_class = eval(optimizer_class) if isinstance(optimizer_class, str) else optimizer_class
    
    
    for s_i, s in enumerate(seeds):
        if (s != None):
            np.random.seed(s)
            torch.manual_seed(s)

        env = gym.make(env, **env_args)
        env.seed(s)
        if isinstance(env.action_space, gym.spaces.Box):
            policy_class.set_bounds(env.action_space.low, env.action_space.high)

        if  policy_class.__name__ == 'TCNPolicy':
            assert not param_based, "The TCNPolicy should be stochastic"
            from utils.state_memory_wrapper import StateMemoryWrapper
            env = StateMemoryWrapper(env)

        if omega is None:
            omega = env.gamma

        # Set behavioural hyper-poliy
        nu_behavioural = hyperpolicy_class(policy=policy_class, policy_args=policy_args, stochastic=param_based, 
                alpha=alpha, beta=beta, state_dim=env.state_dim, omega=omega,  gamma=env.gamma,
                var_bound=var_bound, n_optim_var_bound=n_optim_var_bound, 
                sigma_theta=sigma_theta_behavioural, learn_sigma=learn_sigma, theta_mean=mean_theta_init, 
                use_modulo=use_modulo, #????
                )

        # Set trained hyper-poliy
        nu = hyperpolicy_class(policy=policy_class, policy_args=policy_args, stochastic=param_based, 
                alpha=alpha, beta=beta, state_dim=env.state_dim, omega=omega,  gamma=env.gamma,
                var_bound=var_bound, n_optim_var_bound=n_optim_var_bound, 
                sigma_theta=sigma_theta_init, learn_sigma=learn_sigma, theta_mean=mean_theta_init, 
                use_modulo=use_modulo, #????
                )

        logging.info('\n Hyperpolicy parameters: {}.'.format(nu.get_n_param()))        
        logging.info('\n Policy parameters: {}.'.format(nu.n_policy_params))        
        
        # Set optimizer
        optimizer = optimizer_class(nu.parameters(),  lr=learning_rate,  alpha=0.9, eps=1e-10)

        # Setting variables
        n_optim = np.ceil(n_target_samples/optim_every).astype(int)
        tot_optim = n_optim * epochs_optim

        # Initialiaze variables for statistics 
        params_optim = np.zeros((n_optim, epochs_optim+1, nu.n_params)) if save_extra else None
        thetas_optim = np.zeros((n_optim, epochs_optim, alpha, nu.n_policy_params)) if save_extra else None
        MIS_w_optim = np.zeros((n_optim, epochs_optim, beta, alpha)) if save_extra else None
        states_optim = np.zeros((n_optim, epochs_optim, alpha+1, env.state_dim))
        actions_optim = np.zeros((n_optim, epochs_optim, alpha))
        rewards_optim = np.zeros((n_optim, epochs_optim, alpha))
        J_ahead_optim = np.zeros((n_optim, epochs_optim))
        var_optim = np.zeros((n_optim, epochs_optim))
        loss_optim = np.zeros((n_optim, epochs_optim, 1))

        states = np.zeros((n_behave_samples+n_optim*optim_every+1, env.state_dim))
        thetas = np.zeros((n_behave_samples+n_optim*optim_every, nu.n_policy_params))
        rewards = np.zeros(n_behave_samples+n_optim*optim_every)
        actions = np.zeros(n_behave_samples+n_optim*optim_every)
        ns_process = np.zeros((n_behave_samples+n_optim*optim_every,1))
        info_process = {k:np.zeros(n_behave_samples+n_optim*optim_every) for k in env.info_process}

        # Sample data from behavioural period
        logging.info('Sampling seed {}.'.format(s))
        st, thet, r, a, ns_p, info_p = sampling(nu=nu_behavioural, 
                env=env, n=n_behave_samples, seed=s,)
        
        states[:n_behave_samples+1] = st
        thetas[:n_behave_samples] = thet
        rewards[:n_behave_samples] = r
        actions[:n_behave_samples] = a
        ns_process[:n_behave_samples] = ns_p
        for k,v in info_p.items():
            info_process[k][:n_behave_samples] = v

        # Optimize
        logging.info('Optimize seed {}.'.format(s))        
        for i, t in enumerate(tqdm.tqdm(range(n_behave_samples, n_behave_samples+n_target_samples, optim_every))):
            # Optimization
            nu, optimizer, statistics = optimize(nu=nu, 
                    env=env, optimizer=optimizer, t=t, 
                    state_init=states[t-alpha], grad_steps=grad_steps, 
                    alpha=alpha, beta=beta, lamb_J_behind=lamb_J_behind, lamb_J_ahead=lamb_J_ahead, 
                    lamb_v=lamb_v, grad_samples=grad_samples, 
                    seed=s, show_tqdm=False,)

            states_optim[i] = statistics['states']
            actions_optim[i] = statistics['actions']
            rewards_optim[i] = statistics['rewards']
            J_ahead_optim[i] = statistics['J_ahead']
            var_optim[i] = statistics['variance']
            loss_optim[i] = statistics['losses']
            if save_extra:
                params_optim[i] = statistics['params']
                thetas_optim[i] = statistics['thetas']
                MIS_w_optim[i] = statistics['MIS_weights']

            # Sampling new data
            st, theta, r, a, ns_p, info_p = resume_sampling(env, nu, current_t=t, n_steps=optim_every)
            

            states[t+1:t+optim_every+1] = st
            thetas[t:t+optim_every] = theta
            rewards[t:t+optim_every] = r
            actions[t:t+optim_every] = a
            ns_process[t:t+optim_every] = ns_p
            for k,v in info_p.items():
                info_process[k][t:t+optim_every] = v

        # Save parameters
        save_lifelong(nu, optimizer, params_optim, states_optim, actions_optim, thetas_optim,
                rewards_optim, MIS_w_optim, J_ahead_optim, var_optim, loss_optim, states,
                thetas, rewards, actions, ns_process, info_process, 
                seed=s, save_folder=save_folder)

        # Plot behavioural sampling
        info_process = {k:[np.array(v)] for k,v in info_process.items()}
        logging.info('Save seed {}.'.format(s))
        series = {
            **{k:v[:n_behave_samples] for k,v in info_process.items()},
            'ns process': [ns_process[:n_behave_samples]],
            'states': [states[:n_behave_samples]],
            'thetas': [thetas[:n_behave_samples]],
            'rewards': [rewards[:n_behave_samples]],
            'actions': [actions[:n_behave_samples]]
            }
        save_path = None if (save_folder is None) else os.path.join(save_folder,'behavioural_samples_seed_{}'.format(s))
        plots(series, save_path, title='Env {}'.format(env.unwrapped.spec.id))


        # Plot life-long learning
        series = {
            **info_process,
            'ns process': [ns_process],
            'states': [states],
            'thetas': [thetas],
            'rewards': [rewards],
            'actions': [actions]
            }
        save_path = None if (save_folder is None) else os.path.join(save_folder,'lifelong_learning_seed_{}'.format(s))
        plots(series, save_path, title='Env {}'.format(env.unwrapped.spec.id))


        # Plot last policy
        policy_plot = nu.sample_policy(0)
        series = {
            **{k:[v[0][-alpha:]] for k,v in info_process.items()},
            'states': [states_optim[-1,-1]],
            'rewards': [rewards_optim[-1,-1]],
            'actions': [actions_optim[-1,-1]],
            }
        if save_extra:
            with torch.no_grad():
                theta_mean = nu.theta_mean(torch.arange(t-alpha, t))
            theta_mean = [{'x':theta_mean[:,i], 'label':p} for i,p in enumerate(policy_plot.get_param_names())]
            series['thetas'] = theta_mean,
            series['MIS weights'] = [MIS_w_optim[-1,-1,0,:]],
            series['ns process'] = [{'x':ns_process[-alpha:], 'label':'Reward process'}] + theta_mean,
        else:
            series['ns process'] = [ns_process[-alpha:]]
        save_path = None if (save_folder is None) else os.path.join(save_folder,'last_epoch_samples_params_seed_{}'.format(s))
        plots(series, save_path, title='Env {}'.format(env.unwrapped.spec.id))
        
        # Plot optimization objective
        behave_mean_rewards = np.mean(rewards[:n_behave_samples])
        behave_sum_rewards = np.sum(rewards[:n_behave_samples])
        series = {
            'return (mean)':[{'x':rewards_optim.reshape(tot_optim,-1).mean(1), 'label':'learned'}, 
                    {'x':np.full(tot_optim, behave_mean_rewards), 'label':'Behavioural'}],
            'return (mean)':[rewards_optim.reshape(tot_optim,-1).mean(1)],
            'return (sum)':[{'x':rewards_optim.reshape(tot_optim,-1).sum(1), 'label':'learned'}, 
                    {'x':np.full(tot_optim, behave_sum_rewards), 'label':'Behavioural'}],
            'J2': [J_ahead_optim.reshape(-1)],
            'variance': [var_optim.reshape(-1)],
            'Objective Function ({},{},{})'.format(lamb_J_behind,
                    lamb_J_ahead, lamb_v,): [lamb_J_behind*rewards_optim.reshape(tot_optim,-1).sum(1) + 
                    lamb_J_ahead*J_ahead_optim.reshape(-1) - lamb_v*var_optim.reshape(-1)],
            }
        save_path = None if (save_folder is None) else os.path.join(save_folder,'optimization_objective_seed_{}'.format(s))
        plots(series, save_path, title='Env {}'.format(env.unwrapped.spec.id))


        # Plot parameters learning
        if save_extra:
            params_to_plot = params_optim[:,1:].reshape(tot_optim, params_optim.shape[-1]).T # first param is repeated
            size_param = np.array([0]+nu.get_size_params())
            size_param = np.cumsum(size_param)
            params_to_plot = [[params_to_plot[i:j].T] for i,j in zip(size_param[:-1],size_param[1:])]
            series = {
                **dict(zip(nu.get_name_params(), params_to_plot)),
                }
            save_path = None if (save_folder is None) else os.path.join(save_folder,'optimization_params_seed_{}'.format(s))
            plots(series, save_path, title='Env {}'.format(env.unwrapped.spec.id))

        
    


if __name__ == "__main__":
    fire.Fire()
