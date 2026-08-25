from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from gymnax.environments import spaces
from mujoco_playground import registry

from .common import JaxRLEnv


@dataclass
class PlaygroundInterface(JaxRLEnv):
    env: Any
    env_name: str

    @classmethod
    def make(cls, env_name: str, impl: str = "jax"):
        config_overrides = {"impl": impl}
        env = registry.load(env_name=env_name, config_overrides=config_overrides)
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
        action_space = spaces.Box(
            low=-jnp.ones(action_size, dtype=jnp.float32),
            high=jnp.ones(action_size, dtype=jnp.float32),
            shape=(action_size,),
        )
        return action_space

    @property
    def observation_size(self):
        return self.env.observation_size

    @property
    def action_size(self):
        return self.env.action_size
