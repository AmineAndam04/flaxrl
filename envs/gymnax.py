from dataclasses import dataclass
from typing import Any

import gymnax

from .common import JaxRLEnv


@dataclass
class GymnaxInterface(JaxRLEnv):
    env: Any
    params: Any

    @classmethod
    def make(cls, env_name: str):
        env, params = gymnax.make(env_name)
        return cls(env=env, params=params)

    def reset(self, key):
        return self.env.reset(key, self.params)

    def step(self, key, state, action):
        return self.env.step(key, state, action, self.params)

    @property
    def observation_space(self):
        return self.env.observation_space(self.params)

    @property
    def action_space(self):
        return self.env.action_space(self.params)

    @property
    def observation_size(self):
        obs_space = self.observation_space
        if isinstance(obs_space, gymnax.environments.spaces.Box):
            shape = obs_space.shape
            if len(shape) > 1:
                return shape
            else:
                return shape[0]
        elif isinstance(obs_space, gymnax.environments.spaces.Discrete):
            return obs_space.n

    @property
    def action_size(self):
        action_space = self.env.action_space
        if isinstance(action_space, gymnax.environments.spaces.Box):
            return action_space.shape[0]
        elif isinstance(action_space, gymnax.environments.spaces.Discrete):
            return action_space.n
