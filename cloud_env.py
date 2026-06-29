"""Gymnasium environment for cloud auto-scaling RL (CISC 856)."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from traffic import PoissonTrafficGenerator


class CloudScalingEnv(gym.Env):
    """Cloud cluster auto-scaling env with Discrete(3) actions.
    0 = scale out, 1 = hold, 2 = scale in.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_servers=10, min_servers=1, server_capacity=50,
                 max_queue=500, boot_delay=3, episode_length=1000,
                 traffic_mode="stochastic",
                 reward_weights=(1.0, 0.1, 20.0, 5.0),
                 lambda_max=240.0, seed=None):
        super().__init__()

        self.N_max = max_servers
        self.N_min = min_servers
        self.c = server_capacity
        self.Q_max = max_queue
        self.boot_delay = boot_delay
        self.ep_len = episode_length
        self.traffic_mode = traffic_mode
        self.alpha, self.beta, self.gamma_w, self.delta = reward_weights
        self.lambda_max = lambda_max

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self._rng = np.random.default_rng(seed)
        self.traffic = None

    def _get_obs(self, cpu_util):
        """Build the normalized 5-D observation vector."""
        return np.array([
            self.active / self.N_max,
            len(self.boot_timers) / self.N_max,
            cpu_util,
            self.queue / self.Q_max,
            min(self.arrival_ema, self.lambda_max) / self.lambda_max,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.t = 0
        self.active = 2
        self.boot_timers = []
        self.queue = 0
        self.last_action = 1  # HOLD -- so first thrash check is clean

        self.traffic = PoissonTrafficGenerator(
            seed=int(self._rng.integers(1e9)),
            mode=self.traffic_mode)
        self.arrival_ema = self.traffic.peek_lambda(0)

        obs = self._get_obs(cpu_util=0.0)
        return obs, {"active": self.active, "queue": self.queue}

    def step(self, action):
        # 1. validate action
        applied = action
        provisioned = self.active + len(self.boot_timers)

        if action == 0 and provisioned >= self.N_max:
            applied = 1
        if action == 2 and self.active <= self.N_min:
            applied = 1

        if applied == 0:
            self.boot_timers.append(self.boot_delay)
        elif applied == 2:
            self.active -= 1

        # 2. traffic
        arrivals, current_lambda = self.traffic.generate(self.t)

        # 3. process requests
        capacity = self.active * self.c
        backlog = self.queue + arrivals
        processed = min(backlog, capacity)

        # 4. update queue
        self.queue = max(0, backlog - processed)

        # 5. detect drops
        dropped = max(0, self.queue - self.Q_max)
        self.queue = min(self.queue, self.Q_max)

        # 6. advance boot timers
        still_booting = []
        for timer in self.boot_timers:
            timer -= 1
            if timer <= 0:
                self.active = min(self.N_max, self.active + 1)
            else:
                still_booting.append(timer)
        self.boot_timers = still_booting

        # 7. reward
        cpu_util = min(1.0, backlog / max(1, self.active * self.c))
        thrash = int((self.last_action == 0 and applied == 2) or
                     (self.last_action == 2 and applied == 0))

        C = self.active
        L = (self.queue / self.Q_max) ** 2
        D = dropped
        T = thrash
        reward = -(self.alpha * C + self.beta * L +
                   self.gamma_w * D + self.delta * T)

        # 8. bookkeeping
        self.last_action = applied
        self.arrival_ema = 0.8 * self.arrival_ema + 0.2 * current_lambda
        self.t += 1

        # 9. obs & termination
        obs = self._get_obs(cpu_util).astype(np.float32)
        truncated = self.t >= self.ep_len
        terminated = False  # always -- use truncated for time limits

        info = {
            "dropped": dropped, "active": self.active, "queue": self.queue,
            "cpu_util": cpu_util, "lambda": current_lambda,
            "reward_components": {"C": C, "L": L, "D": D, "T": T},
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        print(f"t={self.t:4d}  active={self.active}  boot={len(self.boot_timers)}  "
              f"queue={self.queue:3d}  lambda={self.arrival_ema:6.1f}")

    def close(self):
        pass


gym.register(id="CloudScaling-v1", entry_point="cloud_env:CloudScalingEnv")


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    print("Running check_env ...")
    env = CloudScalingEnv()
    check_env(env, warn=True)
    print("[OK] check_env passed.\n")

    obs, info = env.reset(seed=42)
    print(f"obs: {obs}")
    print(f"info: {info}\n")

    total_reward = 0.0
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        if truncated:
            break

    print(f"\n20 steps -- total reward: {total_reward:.2f}, "
          f"dropped: {info['dropped']}")
