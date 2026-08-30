from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.environments.spaces import Box, Discrete

from .core import JaxMARLEnv


@dataclass
class SMAXnterface(JaxMARLEnv):
    env: Any
    env_name: str
    num_agents: int
    _action_size: int
    _observation_size: int
    _state_size: int

    @classmethod
    def make(cls, env_name: str, **kwargs):
        scenario = map_name_to_scenario(env_name)
        env = make("HeuristicEnemySMAX", scenario=scenario, **kwargs)
        num_agents = env.num_agents
        act_space = env.action_space(env.agents[0])
        if isinstance(act_space, Discrete):
            _action_size = act_space.n
        elif isinstance(act_space, Box):
            _action_size = act_space.shape[0]
        else:
            raise NotImplementedError(" This env has an action space not yet supported by this interface.")
        _observation_size = env.obs_size
        _state_size = env.state_size
        return cls(
            env=env,
            env_name=env_name,
            num_agents=num_agents,
            _action_size=_action_size,
            _observation_size=_observation_size,
            _state_size=_state_size,
        )

    def reset(self, key) -> tuple[Any, Any]:
        obs, env_state = self.env.reset(key)
        obs, mdp_state = self._process_obs(obs)
        return obs, mdp_state, env_state

    def step(self, key, state, action):
        # TODO write a wrapper that aggregates the rewards
        # TODO recorder wrapper should report battle won
        #! rewards are still an array, not aggregated
        #! dones don't return __all__, only individual dones. It will be added to info
        action = {agent: action[i] for i, agent in enumerate(self.env.agents)}
        obs, env_state, rewards, dones, infos = self.env.step(key=key, state=state, actions=action)
        obs, mdp_state = self._process_obs(obs)
        infos["__all__"] = dones.get("__all__", None)
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
        return self.env.get_avail_actions(state)

    def _process_obs(self, obs):
        obs_ = jnp.array([obs[agent] for agent in self.env.agents])
        mdp_state = obs["world_state"]
        return obs_, mdp_state

    def _dict_to_jnp_array(self, x):
        return jnp.array([x[agent] for agent in self.env.agents])

    @property
    def observation_size(self):
        return self._observation_size

    @property
    def action_size(self):
        return self._action_size

    @property
    def state_size(self):
        return self._state_size
