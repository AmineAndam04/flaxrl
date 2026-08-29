from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

# TODO 1. add Flatten observation


# base wrapper
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


class VecWrapper(Wrapper):
    """Vectorized environments"""

    def __init__(self, env):
        super().__init__(env)

    def reset(self, key):
        return jax.vmap(fun=self.env.reset, in_axes=0)(key)

    def step(self, key, state, action):
        return jax.vmap(fun=self.env.step, in_axes=(0, 0, 0))(key, state, action)

    def sample(self, key):
        return jax.vmap(fun=self.env.action_space.sample, in_axes=0)(key)


## ------ Time Limit wrapper
@struct.dataclass
class TimeLimitState:
    env_state: Any
    elapsed_steps: jax.Array


class TimeLimit(Wrapper):
    """
    Limit the number of environment steps
    """

    def __init__(self, env, max_episode_steps):
        super().__init__(env=env)
        self.max_episode_steps = max_episode_steps

    def reset(self, key):
        obs, env_state = self.env.reset(key)
        new_state = TimeLimitState(
            env_state=env_state, elapsed_steps=jnp.zeros(obs.shape[0], dtype=jnp.int32)
        )
        return obs, new_state

    def step(self, key, state, action):
        obs, env_state, reward, terminated, truncated, info = self.env.step(key, state.env_state, action)
        # truncated
        elapsed_steps = state.elapsed_steps + 1
        time_limit = elapsed_steps >= self.max_episode_steps
        truncated = jnp.logical_or(truncated.astype(bool), time_limit)
        new_state = TimeLimitState(env_state=env_state, elapsed_steps=elapsed_steps)

        return obs, new_state, reward, terminated, truncated, info


## ------ Auto-reset wrapper
@struct.dataclass
class AutoResetState:
    env_state: Any
    cached_state: Any
    cached_obs: Any


class AutoResetWrapper(Wrapper):
    """Auto reset the envs after terminated or truncated"""

    def reset(self, key):
        obs, state = self.env.reset(key)
        new_state = AutoResetState(
            env_state=state, cached_state=jax.tree.map(lambda x: x, state), cached_obs=obs
        )
        return obs, new_state

    def step(self, key, state, action):
        next_obs, env_state, reward, terminated, truncated, info = self.env.step(key, state.env_state, action)
        done = jnp.logical_or(terminated.astype(bool), truncated.astype(bool))

        def where_done(cached, current):
            mask = done.reshape(done.shape + (1,) * (current.ndim - done.ndim))
            return jnp.where(mask, cached, current)

        new_state = jax.tree.map(where_done, state.cached_state, env_state)
        obs = jax.tree.map(where_done, state.cached_obs, next_obs)

        state = AutoResetState(
            env_state=new_state, cached_state=state.cached_state, cached_obs=state.cached_obs
        )
        info = {**info, "final_observation": next_obs}
        return obs, state, reward, terminated, truncated, info


## ------ Normalize observations, valid only for vmap-ed envs
@struct.dataclass
class NormalizeVecObservationState:
    env_state: Any
    mean: jax.Array
    var: jax.Array
    count: float


class NormalizeVecObservation(Wrapper):
    def reset(self, key):
        obs, env_state = self.env.reset(key)
        # compute normalization statistics
        state = NormalizeVecObservationState(
            env_state=env_state, mean=jnp.zeros_like(obs[0]), var=jnp.ones_like(obs[0]), count=1e-4
        )
        new_mean, new_var, new_count = compute_mean_var_count_from_moments(x=obs, normalization_state=state)
        # state
        state = NormalizeVecObservationState(env_state=env_state, mean=new_mean, var=new_var, count=new_count)
        return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state

    def step(self, key, state, action):
        obs, env_state, reward, terminated, truncated, info = self.env.step(key, state.env_state, action)
        # re-compute normalization statistics
        new_mean, new_var, new_count = compute_mean_var_count_from_moments(x=obs, normalization_state=state)
        state = NormalizeVecObservationState(env_state=env_state, mean=new_mean, var=new_var, count=new_count)
        # normalize
        obs = (obs - state.mean) / jnp.sqrt(state.var + 1e-8)
        # normalize final obs
        if "final_observation" in info:
            info = {
                **info,
                "final_observation": (info["final_observation"] - state.mean) / jnp.sqrt(state.var + 1e-8),
            }
        return obs, state, reward, terminated, truncated, info


class NormalizeVecObservationEval(Wrapper):
    def __init__(self, env, obs_mean, obs_var):
        super().__init__(env=env)
        self.obs_mean = obs_mean
        self.obs_var = obs_var

    def reset(self, key):
        obs, env_state = self.env.reset(key)
        return (obs - self.obs_mean) / jnp.sqrt(self.obs_var + 1e-8), env_state

    def step(self, key, state, action):
        obs, env_state, reward, terminated, truncated, info = self.env.step(key, state, action)
        # normalize
        obs = (obs - self.obs_mean) / jnp.sqrt(self.obs_var + 1e-8)
        return obs, env_state, reward, terminated, truncated, info


## ------ Normalize rewards, valid only for vmap-ed envs
@struct.dataclass
class NormalizeVecRewardState:
    env_state: Any
    mean: jax.Array
    var: jax.Array
    count: float
    returns: jax.Array


class NormalizeVecReward(Wrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key):
        obs, env_state = self.env.reset(key)
        state = NormalizeVecRewardState(
            env_state=env_state,
            mean=0.0,
            var=1.0,
            count=1e-4,
            returns=jnp.zeros(key.shape[-1]),
        )
        return obs, state

    def step(self, key, state, action):
        obs, env_state, reward, terminated, truncated, info = self.env.step(key, state.env_state, action)
        # compute normalization statistics
        returns = state.returns * self.gamma * (1 - terminated) + reward
        new_mean, new_var, new_count = compute_mean_var_count_from_moments(
            x=returns, normalization_state=state
        )
        state = NormalizeVecRewardState(
            env_state=env_state, mean=new_mean, var=new_var, count=new_count, returns=returns
        )
        reward = reward / jnp.sqrt(state.var + 1e-8)
        return obs, state, reward, terminated, truncated, info


def compute_mean_var_count_from_moments(x, normalization_state):
    # new batch stats
    batch_mean = jnp.mean(x, axis=0)
    batch_var = jnp.var(x, axis=0)
    batch_count = x.shape[0]

    delta = batch_mean - normalization_state.mean
    new_count = normalization_state.count + batch_count
    # new mean
    new_mean = normalization_state.mean + delta * batch_count / new_count
    # new var
    m_a = normalization_state.var * normalization_state.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * normalization_state.count * batch_count / new_count
    new_var = M2 / new_count
    return new_mean, new_var, new_count


## ------ Record Episodic statistics, valid only for vmap-ed envs
@struct.dataclass
class RecordVecEpisodeStatisticsState:
    env_state: Any
    episode_return: jax.Array
    episode_length: jax.Array


class RecordVecEpisodeStatistics(Wrapper):
    def reset(self, key):
        obs, env_state = self.env.reset(key)
        # start recording
        state = RecordVecEpisodeStatisticsState(
            env_state=env_state,
            episode_return=jnp.zeros(obs.shape[0], dtype=jnp.float32),
            episode_length=jnp.zeros(obs.shape[0], dtype=jnp.int32),
        )
        return obs, state

    def step(self, key, state, action):
        obs, env_state, reward, terminated, truncated, info = self.env.step(key, state.env_state, action)
        done = jnp.logical_or(terminated.astype(bool), truncated.astype(bool))
        # accumulate
        episode_return = state.episode_return + reward
        episode_length = state.episode_length + 1
        # store episodic stats is done
        info = {
            **info,
            "episode_return": jnp.where(done, episode_return, jnp.zeros_like(episode_return)),
            "episode_length": jnp.where(done, episode_length, jnp.zeros_like(episode_length)),
        }
        # drop prev stats if done, otherwise keep accumulating
        state = RecordVecEpisodeStatisticsState(
            env_state=env_state,
            episode_return=jnp.where(done, jnp.zeros_like(episode_return), episode_return),
            episode_length=jnp.where(done, jnp.zeros_like(episode_length), episode_length),
        )
        return obs, state, reward, terminated, truncated, info


## ------ Clip Actions, valid only for "not" vmap-ed envs
class ClipAction(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.low = self.env.action_space.low
        self.high = self.env.action_space.high

    def step(self, key, state, action):
        action = jnp.clip(action, self.low, self.high)
        return self.env.step(key, state, action)
