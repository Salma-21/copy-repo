"""
Agent adapters for the main algorithm comparison.

adapters are used case:
each agent type has a slightly different prediction interface.

Normal SB3 agents such as PPO, A2C, and DQN use:
    action, _ = model.predict(obs, deterministic=True)

Recurrent PPO / PPO-LSTM needs memory:
    action, state = model.predict(obs, state=state, episode_start=episode_start)

The rule-based baseline is not an SB3 model, but it still has predict().

The adapter pattern hides these differences. The evaluator can call every
agent in the same way, without knowing whether the agent is PPO, DQN, A2C,
PPO-LSTM, Sparse PPO, or the baseline.
"""

import numpy as np


class AgentAdapter:
    """Common interface used by the comparison script."""

    def __init__(self, name):
        self.name = name

    def reset_episode(self, num_envs=1):
        """Reset any internal state before a new episode starts."""
        pass

    def predict(self, obs, done):
        """Return the action for the current observation."""
        raise NotImplementedError


class SB3AgentAdapter(AgentAdapter):
    """Adapter for normal Stable-Baselines3 models.

    This works for PPO, A2C, Vanilla DQN, Double DQN, Dueling DQN,Double-Dueling DQN, and Sparse PPO."""

    def __init__(self, name, model):
        super().__init__(name)
        self.model = model

    def predict(self, obs, done):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


class RecurrentPPOAdapter(AgentAdapter):
    """Adapter for PPO-LSTM / Recurrent PPO.

    PPO-LSTM needs to keep an LSTM hidden state across timesteps.
    It also needs episode_start so the LSTM memory is reset correctly
    at the beginning of each episode.
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
    """Adapter for the rule-based baseline.

    The baseline is not trained, but it has a predict() method, so this adapter
    makes it compatible with the same comparison loop.
    """

    def __init__(self, model):
        super().__init__("Rule-Based Baseline")
        self.model = model

    def predict(self, obs, done):
        action, _ = self.model.predict(obs, deterministic=True)
        return action