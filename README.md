# POLIS
Implementation of the Policy Optimization in Lifelong learning
through Importance Sampling (POLIS) algorithm from the paper 
Lifelong Hyper-Policy Optimization with Multiple Importance Sampling
Regularization at AAAI 2022.

The hyper-policy learns 


## Example 
For this example, you must install the local gym-bandits environment. For that, do ```cd env\gym-bandits``` then ```pip install -e .``` .

Then, you can run the following command,
```bash
python run.py run_lifelong --save_folder "results/test" --env_class "PeriodicBandit-v0" --hyperpolicy_class "hypol.SinPolicy" --policy_class "pol.CategoricalPolicy" --sigma_theta_behavioural 0.5 --sigma_theta_init -1 --alpha 300 --n_init_samples 300 --n_optim_samples 500 --optim_every 50 --epochs_optim 100 --seeds 0,1,2 --learning_rate 1e-3 --lamb_J_1 1 --lamb_J_2 1 --lamb_v 2 --beta 300 --grad_replicas 1 --learn_sigma "True" --var_bound="two_step_psi_first"
```


## Cite the paper
@article{liotet2021lifelong,
  title={Lifelong Hyper-Policy Optimization with Multiple Importance Sampling Regularization},
  author={Liotet, Pierre and Vidaich, Francesco and Metelli, Alberto Maria and Restelli, Marcello},
  journal={arXiv preprint arXiv:2112.06625},
  year={2021}
}