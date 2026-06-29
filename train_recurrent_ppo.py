"""
Train PPO with LSTM.

This file trains a Recurrent PPO agent, also called PPO-LSTM.

The normal PPO agent chooses an action from the current observation only.
PPO-LSTM adds memory through an LSTM policy, so the agent can also use
information from recent timesteps.

This matters in cloud autoscaling because the environment is not only about
the current CPU or queue value. Recent traffic trends also matter. For example,
if traffic has been increasing for several steps, the agent may need to scale
out before the queue becomes too large. Server boot delay also makes memory
useful, because a scale-out action does not immediately create a new active
server.

In our project, PPO-LSTM is used to test whether adding memory improves
autoscaling decisions compared with standard PPO, A2C, DQN variants, and the
rule-based baseline.

This script trains the model, evaluates it during training, saves the best and
final policies, saves the normalization statistics, and creates a learning
curve plot.
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# This import registers CloudScaling-v1 so env_factory can create it.
from cloud_env import CloudScalingEnv  # noqa: F401

from env_factory import make_env, make_vec_env
from metrics_callback import MetricsCallback


def plot_eval_curve(eval_path, out_path):
    """Create a plot showing how PPO-LSTM improves during training."""

    if not os.path.exists(eval_path):
        print(f"No eval file found at {eval_path}")
        return

    # EvalCallback stores rewards from periodic evaluations in this file.
    data = np.load(eval_path)
    timesteps = data["timesteps"]
    rewards = data["results"]

    mean_rewards = rewards.mean(axis=1)
    std_rewards = rewards.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, mean_rewards, label="PPO-LSTM mean reward")
    plt.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.25,
    )

    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation reward")
    plt.title("PPO-LSTM Evaluation Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved plot to {out_path}")


def parse_args():
    """Read training settings from the terminal."""

    parser = argparse.ArgumentParser(description="Train PPO-LSTM on CloudScaling-v1")

    parser.add_argument(
        "--timesteps",
        type=int,
        default=2_000_000,
        help="Total number of training timesteps.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for training.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create folders for models, checkpoints, logs, and plots.
    for path in [
        "./models/best_recurrent_ppo",
        "./checkpoints/recurrent_ppo",
        "./logs/recurrent_ppo",
        "./logs/recurrent_ppo_eval",
        "./results/recurrent_ppo",
    ]:
        os.makedirs(path, exist_ok=True)

    # This is the environment used for learning.
    # Multiple copies of the environment let PPO collect larger batches of
    # experience before each policy update.
    train_env = make_vec_env(
        n_envs=8,
        seed=args.seed,
        use_subprocess=False,
        norm_reward=True,
    )

    # This environment is only used for evaluation.
    
    # Reward normalization is disabled so the evaluation reward stays in the real scale of the cloud environment.
    eval_env = VecNormalize(
        DummyVecEnv([make_env(rank=100, seed=args.seed)]),
        norm_obs=True,
        norm_reward=False,
        clip_obs=5.0,
        gamma=0.99,
    )
    eval_env.training = False

    # Define the PPO-LSTM agent.
    # MlpLstmPolicy means the policy has an LSTM memory layer.
    # The LSTM can keep information from previous observations, which helps
    # when traffic changes over time or when server boot delay creates delayed effects.
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            lstm_hidden_size=128,
            n_lstm_layers=1,
            shared_lstm=False,
            enable_critic_lstm=True,
            net_arch=dict(pi=[256], vf=[256]),
        ),
        tensorboard_log="./logs/recurrent_ppo/",
        device=args.device,
        seed=args.seed,
        verbose=1,
    )

    # Evaluate the model during training and save the best version.
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_recurrent_ppo/",
        log_path="./logs/recurrent_ppo_eval/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    # Save backup checkpoints during training.
    ckpt_cb = CheckpointCallback(
        save_freq=100_000,
        save_path="./checkpoints/recurrent_ppo/",
    )

    # Log cloud-specific metrics such as queue length and dropped requests.
    metrics_cb = MetricsCallback()

    print("=" * 60)
    print("[START] PPO-LSTM Training")
    print(f"Device: {args.device} | Timesteps: {args.timesteps:,}")
    print("=" * 60)

    start = time.perf_counter()

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[eval_cb, ckpt_cb, metrics_cb],
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving PPO-LSTM model")

    # Save the final trained model.
    model.save("./models/final_recurrent_ppo")

    # Save normalization statistics so evaluation uses the same observation
    # scaling as training.
    train_env.save("./models/vecnormalize_recurrent_ppo.pkl")

    wall_time = time.perf_counter() - start

    plot_eval_curve(
        eval_path="./logs/recurrent_ppo_eval/evaluations.npz",
        out_path="./results/recurrent_ppo/recurrent_ppo_eval_curve.png",
    )

    print("=" * 60)
    print("[DONE] PPO-LSTM Training")
    print(f"Wall time: {wall_time:.1f}s")
    print("Final model: ./models/final_recurrent_ppo.zip")
    print("Best model:  ./models/best_recurrent_ppo/best_model.zip")
    print("VecNormalize: ./models/vecnormalize_recurrent_ppo.pkl")
    print("=" * 60)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()