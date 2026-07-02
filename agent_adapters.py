"""Adapters for evaluating different policy APIs with one loop."""

import numpy as np


class AgentAdapter:
    """Base adapter interface used by evaluation scripts."""

    def __init__(self, name):
        self.name = name

    def reset_episode(self, num_envs=1):
        """Reset any per-episode state."""

    def predict(self, obs, done):
        raise NotImplementedError


class SB3AgentAdapter(AgentAdapter):
    """Adapter for normal SB3 models: PPO, A2C, and DQN variants."""

    def __init__(self, name, model):
        super().__init__(name)
        self.model = model

    def predict(self, obs, done=None):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


class RecurrentPPOAdapter(AgentAdapter):
    """Adapter for PPO-LSTM / RecurrentPPO."""

    def __init__(self, name, model):
        super().__init__(name)
        self.model = model
        self.lstm_state = None
        self.episode_start = None

    def reset_episode(self, num_envs=1):
        self.lstm_state = None
        self.episode_start = np.ones((num_envs,), dtype=bool)

    def predict(self, obs, done=None):
        action, self.lstm_state = self.model.predict(
            obs,
            state=self.lstm_state,
            episode_start=self.episode_start,
            deterministic=True,
        )
        self.episode_start = done
        return action


class NoScalingBaselineAdapter(AgentAdapter):
    """Non-learning baseline that always chooses the no-change action."""

    def __init__(self, action=1):
        super().__init__("No-Scaling Baseline")
        self.action = action
        self.num_envs = 1

    def reset_episode(self, num_envs=1):
        self.num_envs = num_envs

    def reset(self):
        self.reset_episode(1)

    def predict(self, obs, done=None):
        return np.full((self.num_envs,), self.action, dtype=np.int64)


class BaselineAdapter(AgentAdapter):
    """Generic wrapper for any baseline object that has predict()."""

    def __init__(self, name, model):
        super().__init__(name)
        self.model = model

    def predict(self, obs, done=None):
        action, _ = self.model.predict(obs, deterministic=True)
        return action