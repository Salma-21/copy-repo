"""
Adapter classes for comparing different agents with one evaluation loop.

we use adapters case not all agents are called in the same way.

Normal SB3 models such as PPO, A2C, and DQN use:
    model.predict(obs, deterministic=True)

PPO-LSTM is different because it needs memory:
    model.predict(obs, state=lstm_state, episode_start=episode_start)

The rule-based baseline is also different because it is not a trained SB3 model.

The adapter pattern hides these differences. The evaluator only calls:
    adapter.reset_episode()
    adapter.predict(obs, done)

So our main comparison code stays clean.
"""

import numpy as np


class AgentAdapter:
    def __init__(self, name):
        self.name = name

    def reset_episode(self, num_envs=1):
        pass

    def predict(self, obs, done):
        raise NotImplementedError


class SB3AgentAdapter(AgentAdapter):
    """Adapter for normal SB3 models: PPO, A2C, DQN variants, and Sparse PPO."""

    def __init__(self, name, model):
        super().__init__(name)
        self.model = model

    def predict(self, obs, done):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


class RecurrentPPOAdapter(AgentAdapter):
    """this is adapter for PPO-LSTM. PPO-LSTM needs to keep its LSTM state during the episode.
    """

    def __init__(self, name, model):
        super().__init__(name)
        self.model = model
        self.lstm_state = None
        self.episode_start = None

    def reset_episode(self, num_envs=1):
        self.lstm_state = None
        self.episode_start = np.ones((num_envs,), dtype=bool)

    def predict(self, obs, done):
        action, self.lstm_state = self.model.predict(
            obs,
            state=self.lstm_state,
            episode_start=self.episode_start,
            deterministic=True,
        )
        self.episode_start = done
        return action


class BaselineAdapter(AgentAdapter):
    """Adapter for the rule-based baseline."""

    def __init__(self, model):
        super().__init__("Rule-Based Baseline")
        self.model = model

    def predict(self, obs, done):
        action, _ = self.model.predict(obs, deterministic=True)
        return action