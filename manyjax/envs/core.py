from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# TODO specify the structure of outputs: -> tuple[Any, Any, Any, Any]


class JaxRLEnv(ABC):
    @abstractmethod
    def reset(self, key):
        """Should return: obs, state"""
        pass

    @abstractmethod
    def step(self, key, state, action):
        """Should return : obs, state, reward, terminated, truncated, info"""
        pass

    @property
    @abstractmethod
    def observation_space(self): ...

    @property
    @abstractmethod
    def action_space(self): ...

    @property
    @abstractmethod
    def observation_size(self): ...

    @property
    @abstractmethod
    def action_size(self): ...

    @property
    def unwrapped(self):
        return self


class JaxMARLEnv(ABC):
    @abstractmethod
    def reset(self, key) -> tuple[Any, Any, Any, Any]:
        """Should return: obs, state"""
        pass

    @abstractmethod
    def step(self, key, state, action):
        """Should return : obs, mdp_state,env_state, reward, terminated, truncated, info"""
        pass

    @abstractmethod
    def sample(key): ...

    @abstractmethod
    def get_avail_actions(self, state): ...

    @property
    @abstractmethod
    def observation_size(self): ...

    @property
    @abstractmethod
    def action_size(self): ...

    @property
    @abstractmethod
    def state_size(self): ...

    @property
    @abstractmethod
    def reward_size(self): ...

    @property
    def unwrapped(self):
        return self
