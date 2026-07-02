"""Adapters for evaluating different policy APIs with one loop."""

import numpy as np


class AgentAdapter:
    """Base adapter interface used by evaluation scripts."""

    def __init__(self, name):
        self.name = name

    def reset_episode(self, num_envs=1):
        pass

    def predict(self, obs, done=None):
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


class BaselineAdapter(AgentAdapter):
    """Adapter for RuleBasedBaseline from baseline_agent.py."""

    def __init__(self, name, model):
        super().__init__(name)
        self.model = model

    def predict(self, obs, done=None):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
