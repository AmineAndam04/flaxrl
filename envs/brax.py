from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from brax import envs
from gymnax.environments import spaces

from .common import JaxRLEnv


@dataclass
class BraxInterface(JaxRLEnv):
    env: Any
    env_name: str

    @classmethod
    def make(cls, env_name: str, backend: str = "generalized"):
        env = envs.get_environment(env_name=env_name, backend=backend)
        return cls(env=env, env_name=env_name)

    def reset(self, key) -> tuple[Any, Any]:
        state = self.env.reset(rng=key)
        return state.obs, state

    def step(self, key, state, action):
        state = self.env.step(state=state, action=action)
        info = {**state.metrics, **state.info}
        return state.obs, state, state.reward, state.done, False, info

    @property
    def observation_space(self):
        obs_size = self.env.observation_size
        obs = jnp.inf * jnp.ones(obs_size)
        observation_space = spaces.Box(low=-obs, high=obs, shape=(obs_size,))
        return observation_space

    @property
    def action_space(self):
        action_size = self.env.action_size
        action = self.env.sys.actuator.ctrl_range
        action_space = spaces.Box(low=action[:, 0], high=action[:, 1], shape=(action_size,))
        return action_space

    @property
    def observation_size(self):
        return self.env.observation_size

    @property
    def action_size(self):
        return self.env.action_size
