from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jaxmarl import make
from jaxmarl.environments.spaces import Box, Discrete

from .core import JaxMARLEnv


@dataclass
class MPEInterface(JaxMARLEnv):
    env: Any
    env_name: str
    num_agents: int
    action_sizes: list
    longest_action_size: int
    longest_observation_size: int
    _state_size: int
    is_discrete_action: bool
    avail_actions: jnp.ndarray

    @classmethod
    def make(cls, env_name: str):
        env = make(env_id=env_name)
        num_agents = env.num_agents
        if isinstance(env.action_space(env.agents[0]), Discrete):
            action_sizes = [act_space.n for act_space in env.action_spaces.values()]
            longest_action_size = max(action_sizes)
            is_discrete_action = True
        elif isinstance(env.action_space(env.agents[0]), Box):
            action_sizes = [obs_space.shape[0] for obs_space in env.action_spaces.values()]
            longest_action_size = max(action_sizes)
            is_discrete_action = False
        else:
            raise NotImplementedError(" This env has an action space not yet supported by this interface.")
        observation_sizes = [obs_space.shape[0] for obs_space in env.observation_spaces.values()]
        longest_observation_size = max(observation_sizes)
        _state_size = sum(observation_sizes)
        # compute action mask once for all
        avail_actions = jnp.zeros((num_agents, longest_action_size))
        for i in range(num_agents):
            avail_actions = avail_actions.at[i, 0 : action_sizes[i]].set(1)
        avail_actions = avail_actions.astype(bool)
        return cls(
            env=env,
            env_name=env_name,
            num_agents=num_agents,
            action_sizes=action_sizes,
            longest_action_size=longest_action_size,
            longest_observation_size=longest_observation_size,
            _state_size=_state_size,
            is_discrete_action=is_discrete_action,
            avail_actions=avail_actions,
        )

    def reset(self, key):
        obs, env_state = self.env.reset(key)
        obs, mdp_state = self._process_obs(obs)
        return obs, mdp_state, env_state

    def step(self, key, state, action):
        # TODO write a wrapper that aggregates the rewards
        #! rewards are still an array, not aggregated
        #! dones don't return __all__, only individual dones. It will be added to info
        if self.is_discrete_action:
            action = {agent: action[i] for i, agent in enumerate(self.env.agents)}
        else:
            action = {agent: action[i, : self.action_sizes[i]] for i, agent in enumerate(self.env.agents)}
        obs, env_state, rewards, dones, infos = self.env.step(key=key, state=state, actions=action)
        obs, mdp_state = self._process_obs(obs)
        infos = {**infos, "__all__": dones["__all__"]}
        rewards = self._dict_to_jnp_array(rewards)
        dones = self._dict_to_jnp_array(dones)
        return obs, mdp_state, env_state, rewards, dones, False, infos

    def sample(self, key):
        ## TODO  should these functions be vmaped, or is it enough to do so in the training script
        sample_keys = jax.random.split(key, self.num_agents)
        action = jnp.array(
            [
                self.env.action_spaces[self.env.agents[i]].sample(sample_keys[i])
                for i in range(self.num_agents)
            ]
        )
        return action

    def get_avail_actions(self, state):
        return self.avail_actions

    def _process_obs(self, obs):
        obs_ = [obs[agent] for agent in self.env.agents]
        obs = jnp.array(
            [jnp.concat([obs_i, jnp.zeros(self.longest_observation_size - obs_i.shape[0])]) for obs_i in obs_]
        )
        mdp_state = jnp.concat(obs_)
        return obs, mdp_state

    def _dict_to_jnp_array(self, x):
        return jnp.array([x[agent] for agent in self.env.agents])

    @property
    def observation_size(self):
        return self.longest_observation_size

    @property
    def action_size(self):
        return self.longest_action_size

    @property
    def reward_size(self):
        return (self.num_agents,)

    @property
    def state_size(self):
        return self._state_size
