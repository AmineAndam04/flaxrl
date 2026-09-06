from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from gymnax.environments import spaces
from mujoco_playground import registry

from .core import JaxRLEnv


@dataclass
class PlaygroundInterface(JaxRLEnv):
    env: Any
    env_name: str
    _get_obs: Any
    _obs_size: int

    @classmethod
    def make(cls, env_name: str, impl: str = "jax", which_obs="state"):
        config_overrides = {"impl": impl}
        env = registry.load(env_name=env_name, config_overrides=config_overrides)
        obs_size = env.observation_size

        if isinstance(obs_size, Mapping):
            assert which_obs in obs_size
            _get_obs = lambda x: x[which_obs]
            _obs_size = obs_size[which_obs][0]
        else:
            _get_obs = lambda x: x
            _obs_size = obs_size

        return cls(env=env, env_name=env_name, _get_obs=_get_obs, _obs_size=_obs_size)

    def reset(self, key) -> tuple[Any, Any]:
        state = self.env.reset(rng=key)
        return self._get_obs(state.obs), state

    def step(self, key, state, action):
        state = self.env.step(state=state, action=action)
        info = {**state.metrics, **state.info}
        return self._get_obs(state.obs), state, state.reward, state.done, False, info

    @property
    def observation_space(self):
        obs_size = self._obs_size
        obs = jnp.inf * jnp.ones(obs_size)
        observation_space = spaces.Box(low=-obs, high=obs, shape=(obs_size,))
        return observation_space

    @property
    def action_space(self):
        action_size = self.env.action_size
        action_space = spaces.Box(
            low=-jnp.ones(action_size, dtype=jnp.float32),
            high=jnp.ones(action_size, dtype=jnp.float32),
            shape=(action_size,),
        )
        return action_space

    @property
    def observation_size(self):
        return self._obs_size

    @property
    def action_size(self):
        return self.env.action_size
