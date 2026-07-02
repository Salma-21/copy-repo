"""Small adapter layer for evaluating different agent APIs consistently.

Most Stable-Baselines3 models use:
    model.predict(obs, deterministic=True)

Recurrent PPO/PPO-LSTM also needs the LSTM state and an episode-start flag.
The comparison script should not care about those API differences, so each
agent is wrapped behind reset_episode() and predict().
"""

import numpy as np


class AgentAdapter:
    """Base adapter interface used by the evaluation loop."""

    def __init__(self, name):
        self.name = name

    def reset_episode(self, num_envs=1):
        """Reset per-episode state before evaluating one episode."""

    def predict(self, obs, done):
        raise NotImplementedError


class SB3AgentAdapter(AgentAdapter):
    """Adapter for non-recurrent SB3 models: PPO, A2C, and DQN variants."""

    def __init__(self, name, model):
        super().__init__(name)
        self.model = model

    def predict(self, obs, done):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


class RecurrentPPOAdapter(AgentAdapter):
    """Adapter for PPO-LSTM/RecurrentPPO models."""

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
    """Adapter for the rule-based baseline, if included in a future run."""

    def __init__(self, model):
        super().__init__("Rule-Based Baseline")
        self.model = model

    def predict(self, obs, done):
        action, _ = self.model.predict(obs, deterministic=True)
        return action