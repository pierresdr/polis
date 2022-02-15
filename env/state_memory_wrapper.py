


import numpy as np
from gym import spaces
from gym import Wrapper


class StateMemoryWrapper(Wrapper):
    """Converts the image observation from RGB to gray scale.
    """

    def __init__(self, env, memory_size=10,):
        super(StateMemoryWrapper, self).__init__(env)
        self.memory_size = memory_size
        high = self.observation_space.high.reshape(1,-1).repeat(memory_size)
        low = self.observation_space.low.reshape(1,-1).repeat(memory_size)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.state_dim = self.env.state_dim * memory_size
        self.real_state_dim = self.env.state_dim


    def reset(self, **kwargs):
        # Reset the underlying Environment
        obs = self.env.reset(**kwargs)
        self.true_state = obs

        self.state = [obs for _ in range(self.memory_size)]
        return self.get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.real_state = obs
        del self.state[0]
        self.state.append(obs)
        return self.get_obs(), reward, done, info

    def get_obs(self):
        return np.array(self.state).reshape(-1)

    def replay_step(self, states, actions, t):
        states = states.reshape(-1, self.memory_size, self.real_state_dim)
        obs, reward, done, info = self.env.replay_step(states[:,-1], actions, t)

        states = np.roll(states, -1, axis=0)
        states[:,-1] = obs
        states = states.reshape(-1, self.memory_size*self.real_state_dim)
        return states, reward, done, info