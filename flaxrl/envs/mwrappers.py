from typing import Any

import jax
import jax.numpy as jnp
from flax import struct


class Wrapper:
    def __init__(self, env):
        self.env = env

    def reset(self, key):
        return self.env.reset(key)

    def step(self, key, state, action):
        return self.env.step(key, state, action)

    def __getattr__(self, name):
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)


class VecMARLWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, key):
        return jax.vmap(fun=self.env.reset, in_axes=0)(key)

    def step(self, key, state, action):
        return jax.vmap(fun=self.env.step, in_axes=(0, 0, 0))(key, state, action)

    def get_avail_actions(self, state):
        return jax.vmap(fun=self.env.get_avail_actions, in_axes=0)(state)

    def sample(self, key, state):
        return jax.vmap(self.env.sample, in_axes=0)(key, state)


#! JAXmarl don't need a TimeLimit or Auto-reset wrapper
## ------ Record Episodic statistics, valid only for vmap-ed envs


@struct.dataclass
class RecordVecEpisodeStatisticsState:
    env_state: Any
    episode_return: jax.Array
    episode_length: jax.Array


class RecordVecMARLEpisodeStatistics(Wrapper):
    def reset(self, key):
        obs, mdp_state, env_state = self.env.reset(key)
        # start recording
        state = RecordVecEpisodeStatisticsState(
            env_state=env_state,
            episode_return=jnp.zeros((obs.shape[0], *self.reward_size), dtype=jnp.float32),
            episode_length=jnp.zeros((obs.shape[0], *self.reward_size), dtype=jnp.int32),
        )
        return obs, mdp_state, state

    def step(self, key, state, action):
        obs, mdp_state, env_state, rewards, dones, truncated, infos = self.env.step(
            key, state.env_state, action
        )
        done = infos["__all__"].astype(bool)
        done = done.reshape((done.shape[0],) + (1,) * len(self.reward_size))
        # accumulate
        episode_return = state.episode_return + rewards
        episode_length = state.episode_length + 1
        # store episodic stats is done
        infos = {
            **infos,
            "episode_return": jnp.where(done, episode_return, jnp.zeros_like(episode_return)),
            "episode_length": jnp.where(done, episode_length, jnp.zeros_like(episode_length)),
        }
        # drop prev stats if done, otherwise keep accumulating
        state = RecordVecEpisodeStatisticsState(
            env_state=env_state,
            episode_return=jnp.where(done, jnp.zeros_like(episode_return), episode_return),
            episode_length=jnp.where(done, jnp.zeros_like(episode_length), episode_length),
        )
        return obs, mdp_state, state, rewards, dones, truncated, infos

    def sample(self, key, state):
        return self.env.sample(key, state.env_state)

    def get_avail_actions(self, state):
        return self.env.get_avail_actions(state.env_state)


class RewardAggregatorWrapper(Wrapper):
    def __init__(self, env, reward_aggr="mean"):
        super().__init__(env)
        self.reward_aggr = reward_aggr
        if reward_aggr not in ("mean", "sum", "none"):
            raise ValueError(f"{self.reward_aggr} is not supported")

    def step(self, key, state, action):
        obs, mdp_state, env_state, rewards, dones, truncated, infos = self.env.step(key, state, action)
        if self.reward_aggr == "mean":
            rewards = jnp.mean(rewards, axis=-1, keepdims=True)
            dones = jnp.expand_dims(infos["__all__"], axis=1)
            truncated = jnp.expand_dims(truncated, axis=1)
        elif self.reward_aggr == "sum":
            rewards = jnp.sum(rewards, axis=-1, keepdims=True)
            dones = jnp.expand_dims(infos["__all__"], axis=1)
            truncated = jnp.expand_dims(truncated, axis=1)
        elif self.reward_aggr == "none":
            truncated = jnp.repeat(jnp.expand_dims(truncated, axis=1), self.env.num_agents, -1)
        return obs, mdp_state, env_state, rewards, dones, truncated, infos

    @property
    def reward_size(self):
        if self.reward_aggr in ("mean", "sum"):
            return (1,)
        else:
            return (self.env.num_agents,)


class AgentID(Wrapper):
    def reset(self, key):
        obs, mdp_state, env_state = self.env.reset(key)
        obs = self._add_agent_ids(obs)
        return obs, mdp_state, env_state

    def step(self, key, state, action):
        obs, mdp_state, env_state, rewards, dones, truncated, infos = self.env.step(key, state, action)
        obs = self._add_agent_ids(obs)
        return obs, mdp_state, env_state, rewards, dones, truncated, infos

    def _add_agent_ids(self, obs):
        agent_ids = jnp.eye(self.num_agents)
        agent_ids = jnp.broadcast_to(agent_ids, (obs.shape[0], *agent_ids.shape))
        obs = jnp.concat([obs, agent_ids], axis=-1)
        return obs

    @property
    def observation_size(self):
        return self.env.observation_size + self.env.num_agents
