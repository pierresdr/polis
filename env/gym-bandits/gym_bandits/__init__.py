from gym.envs.registration import register

register(
    id='BanditUNRateMA-v0',
    entry_point='gym_bandits.envs:BanditUNRateMA',
)

register(
    id='VasicekBandit-v0',
    entry_point='gym_bandits.envs:VasicekBandit',
)

register(
    id='PeriodicBandit-v0',
    entry_point='gym_bandits.envs:PeriodicBandit',
)

register(
    id='DriftBandit-v0',
    entry_point='gym_bandits.envs:DriftBandit',
)

register(
    id='DriftSinBandit-v0',
    entry_point='gym_bandits.envs:DriftSinBandit',
)


