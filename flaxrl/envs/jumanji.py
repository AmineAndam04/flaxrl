from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jumanji
import numpy as np
from flax import struct
from jumanji.specs import MultiDiscreteArray
from jumanji.wrappers import AutoResetWrapper

from .core import JaxMARLEnv


@struct.dataclass
class JumanjiState:
    env_state: Any
    action_mask: jax.Array


def cleaner_obs_state_size(observation_spec, obs_processor="flatten"):
    # obs has a grid and agent locations
    grid_shape = observation_spec.grid.shape
    loc_shape = observation_spec.agents_locations.shape
    if obs_processor == "flatten":
        return int(np.prod(grid_shape) + loc_shape[-1]), int(np.prod(grid_shape) + np.prod(loc_shape))
    else:
        raise ValueError(" We only support flattened observations")


def connector_obs_state_size(observation_spec, obs_processor="flatten"):
    # obs has a grid and agent locations
    # TODO the env_state has some infos that can be used as a state
    grid_shape = observation_spec.grid.shape
    if obs_processor == "flatten":
        return int(np.prod(grid_shape)), int(np.prod(grid_shape))
    else:
        raise ValueError(" We only support flattened observations")


def lbf_obs_state_size(observation_spec, obs_processor="flatten"):
    # obs has a grid and agent locations
    # TODO the env_state has some infos that can be used as a state
    grid_shape = observation_spec.agents_view.shape
    if obs_processor == "flatten":
        return int(grid_shape[-1]), int(np.prod(grid_shape))
    else:
        raise ValueError(" We only support flattened observations")


def rware_obs_state_size(observation_spec, obs_processor="flatten"):
    # obs has a grid and agent locations
    # TODO the env_state has some infos that can be used as a state
    grid_shape = observation_spec.agents_view.shape
    if obs_processor == "flatten":
        return int(grid_shape[-1]), int(np.prod(grid_shape))
    else:
        raise ValueError(" We only support flattened observations")


def cleaner_get_obs_mdp_state(timestep):
    grid = timestep.observation.grid
    agents_locations = timestep.observation.agents_locations
    grid = jnp.repeat(grid.reshape(-1)[None, :], timestep.observation.agents_locations.shape[0], axis=0)
    obs = jnp.concat([grid, agents_locations], axis=-1)
    mdp_state = jnp.concat(
        [timestep.observation.grid.reshape(-1), timestep.observation.agents_locations.reshape(-1)]
    )
    return obs, mdp_state


def connector_get_obs_mdp_state(timestep):
    obs = jnp.repeat(timestep.observation.grid.reshape(-1)[None, :], timestep.reward.shape[0], axis=0)
    mdp_state = timestep.observation.grid.reshape(-1)
    return obs, mdp_state


def lbf_get_obs_mdp_state(timestep):
    obs = timestep.observation.agents_view
    mdp_state = timestep.observation.agents_view.reshape(-1)
    return obs, mdp_state


def rware_get_obs_mdp_state(timestep):
    obs = timestep.observation.agents_view
    mdp_state = timestep.observation.agents_view.reshape(-1)
    return obs, mdp_state


def cleaner_reward_dones(timestep):
    num_agents = timestep.observation.agents_locations.shape[0]
    rewards = jnp.repeat(timestep.reward, num_agents)
    terminated = jnp.repeat(~timestep.discount.astype(bool), num_agents)
    truncated = timestep.last().astype(bool)
    return rewards, terminated, truncated


def connector_reward_dones(timestep):
    rewards = timestep.reward
    terminated = ~timestep.discount.astype(bool)
    truncated = timestep.last().astype(bool)
    return rewards, terminated, truncated


def lbf_reward_dones(timestep):
    rewards = timestep.reward
    terminated = ~timestep.discount.astype(bool)
    truncated = timestep.last().astype(bool)
    return rewards, terminated, truncated


def rware_reward_dones(timestep):
    num_agents = timestep.observation.action_mask.shape[0]
    rewards = jnp.repeat(timestep.reward, num_agents)
    terminated = jnp.repeat(~timestep.discount.astype(bool), num_agents)
    truncated = timestep.last().astype(bool)
    return rewards, terminated, truncated


FUNC_OBS_STATE_SIZE = {
    "Cleaner-v0": cleaner_obs_state_size,
    "Connector-v3": connector_obs_state_size,
    "RobotWarehouse-v0": rware_obs_state_size,
    "LevelBasedForaging-v0": lbf_obs_state_size,
}
FUNC_OBS_STATE = {
    "Cleaner-v0": cleaner_get_obs_mdp_state,
    "Connector-v3": connector_get_obs_mdp_state,
    "LevelBasedForaging-v0": lbf_get_obs_mdp_state,
    "RobotWarehouse-v0": rware_get_obs_mdp_state,
}
FUNC_REWARD_DONES = {
    "Cleaner-v0": cleaner_reward_dones,
    "Connector-v3": connector_reward_dones,
    "LevelBasedForaging-v0": lbf_reward_dones,
    "RobotWarehouse-v0": rware_reward_dones,
}


@dataclass
class JumanjInterface(JaxMARLEnv):
    env: Any
    env_name: str
    num_agents: int
    _action_size: int
    _observation_size: int
    _state_size: int
    _get_obs_mdp_state: Any
    _get_reward_dones: Any

    @classmethod
    def make(cls, env_name: str, obs_processor="flatten"):
        env = jumanji.make(env_name)
        num_agents = env.num_agents
        if isinstance(env.action_spec, MultiDiscreteArray):
            _action_size = int(env.action_spec.num_values[0])
        else:
            raise NotImplementedError(" This env has an action space not yet supported by this interface.")
        _observation_size, _state_size = FUNC_OBS_STATE_SIZE[env_name](
            observation_spec=env.observation_spec, obs_processor=obs_processor
        )
        _get_obs_mdp_state = jax.jit(FUNC_OBS_STATE[env_name])
        _get_reward_dones = jax.jit(FUNC_REWARD_DONES[env_name])
        env = AutoResetWrapper(env)
        return cls(
            env=env,
            env_name=env_name,
            num_agents=num_agents,
            _action_size=_action_size,
            _observation_size=_observation_size,
            _state_size=_state_size,
            _get_obs_mdp_state=_get_obs_mdp_state,
            _get_reward_dones=_get_reward_dones,
        )

    def reset(self, key):
        env_state_, timestep = self.env.reset(key)
        obs, mdp_state = self._get_obs_mdp_state(timestep)
        env_state = JumanjiState(env_state=env_state_, action_mask=timestep.observation.action_mask)
        return obs, mdp_state.astype(jnp.float32), env_state

    def step(self, key, state, action):
        env_state_, timestep = self.env.step(state.env_state, action)
        obs, mdp_state = self._get_obs_mdp_state(timestep)
        rewards, terminated, truncated = self._get_reward_dones(timestep)
        infos = timestep.extras
        infos = {**infos, "__all__": jnp.all(truncated | terminated)}
        env_state = JumanjiState(env_state=env_state_, action_mask=timestep.observation.action_mask)
        return obs, mdp_state.astype(jnp.float32), env_state, rewards, terminated, truncated, infos

    def sample(self, key, state):
        avail_actions = self.get_avail_actions(state)
        logits = jnp.where(avail_actions, 0.0, -jnp.inf)
        return jax.random.categorical(key, logits, axis=-1)

    def get_avail_actions(self, state):
        return state.action_mask

    @property
    def observation_size(self):
        return self._observation_size

    @property
    def action_size(self):
        return self._action_size

    @property
    def reward_size(self):
        return (self.num_agents,)

    @property
    def state_size(self):
        return self._state_size
