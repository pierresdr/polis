
import numpy as np
import torch
import multiprocessing as mp


def sampling(nu, env, n, seed=None,):
    s = np.zeros((n+1, env.state_dim))
    thet = np.zeros((n, nu.n_policy_params))
    r = np.zeros(n)
    a = np.zeros(n)
    non_stat_process  = np.zeros((n, env.ns_dim))
    info_process = {k:[] for k in env.info_process}
    
    s[0] = env.reset()
    for t in range(n):
        # Sample policy
        policy = nu.sample_policy(t)
        thet[t] = policy.theta

        # Sample action
        sampled_a = policy.sample_action(env.state)

        # Step
        s[t+1], r[t], done, info = env.step(sampled_a)
        a[t] = sampled_a

        # Save env information
        for k,v in info.items():
            info_process[k].append(v)

        non_stat_process[t] = env.ns_history[t]
    return s, thet, r, a, non_stat_process, info_process


def replay(nu, env, t, state_init, alpha,):
    assert len(env.ns_history)>=alpha, 'Inconsistent length of contexts array'
    
    thet = np.zeros((alpha, nu.n_policy_params))
    r = np.zeros(alpha)
    a = np.zeros(alpha)
    s = np.zeros((alpha+1, env.state_dim))
    info_p = {k:np.zeros(alpha) for k in env.info_process}
    s[0] = state_init
    
    for i, t_i in enumerate(range(t-alpha, t)):
        # Sample policy
        policy = nu.sample_policy(t_i)
        thet[i] = policy.theta

        # Sample action
        sampled_a = policy.sample_action(s[i])

        # Step
        s[i+1], r[i], done, info = env.replay_step(s[i], sampled_a, t_i)
        a[i] = sampled_a

        # Save env information
        for k,v in info.items():
            info_p[k][i] = v
    return s, r, thet, a, info_p


def replay_parallel(nu, env, t, state_init, alpha, n_play=2):
    assert len(env.ns_history)>=alpha, 'Inconsistent length of contexts array'
    
    r = np.zeros((n_play, alpha))
    a = np.zeros((n_play, alpha))
    s = np.zeros((n_play, alpha+1, env.state_dim))
    info_p = {k:np.zeros((n_play, alpha)) for k in env.info_process}
    s[:,0] = state_init
    
    if nu.policy.__name__=='TCNPolicy':
        # Sample policies for all the period in parallel
        policy, thet = nu.sample_policy(torch.arange(t-alpha+1, t+1), n_policies=1)
        thet = thet.repeat(n_play, 0)
        for i, t_i in enumerate(range(t-alpha+1, t+1)):
            # Sample action
            sampled_a = policy[0].sample_action_parallel(s[:,i],)

            #Step
            s[:,i+1], r[:,i], done, info = env.replay_step(s[:,i], sampled_a, t_i)
            a[:,i] = sampled_a

            # Save env information
            for k,v in info.items():
                info_p[k][:,i] = v
    else:
        # Sample policies for all the period in parallel
        policies, thet = nu.sample_policy(torch.arange(t-alpha+1, t+1), n_policies=n_play)
        for i, t_i in enumerate(range(t-alpha+1, t+1)):
            # Sample action
            sampled_a = np.array([p.sample_action(s[p_i, i]) for p_i, p in enumerate(policies[i*n_play:(i+1)*n_play])])
            
            #Step
            s[:,i+1], r[:,i], done, info = env.replay_step(s[:,i], sampled_a, t_i)
            a[:,i] = sampled_a

            # Save env information
            for k,v in info.items():
                info_p[k][:,i] = v

    return s, r, thet.astype(float), a, info_p





def resume_sampling(env, nu, current_t, n_steps):
    s = np.zeros((n_steps, env.state_dim))
    thet = np.zeros((n_steps, nu.n_policy_params))
    r = np.zeros(n_steps)
    a = np.zeros(n_steps)
    ns_process  = np.zeros((n_steps, env.ns_dim))
    info_process = {k:[] for k in env.info_process}

    for t_i, t in enumerate(range(current_t, current_t+n_steps)):
        # Sample policy parameters
        policy = nu.sample_policy(t)
        thet[t_i] = policy.theta
        
        # Sample action
        sampled_a = policy.sample_action(env.state)

        # Step
        s[t_i], r[t_i], done, info = env.step(sampled_a)
        a[t_i] = sampled_a
        
        # Save env information
        ns_process[t_i] = env.ns_history[t]
        for k,v in info.items():
            info_process[k].append(v)

    return s, thet, r, a, ns_process, info_process