"""
Train PPO-LSTM variants on CloudScaling-v1.

we keeps the original PPO-LSTM training structure, but adds a variant
option so different recurrent PPO settings can be tested cleanly.

The goal is to improve PPO-LSTM's robustness. In the previous results,
PPO-LSTM was cost-efficient, but it became risky under bursty spike traffic.
These variants test whether lower learning rate, smaller PPO updates, larger
LSTM memory, or harder spike training can make the recurrent policy more stable.
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cloud_env import CloudScalingEnv  # noqa: F401
from env_factory import make_env, make_vec_env
from metrics_callback import MetricsCallback


LSTM_VARIANTS = {
    # Similar to the original PPO-LSTM setup.
    "balanced": dict(
        learning_rate=3e-4,
        n_steps=1024,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.01,
        lstm_hidden_size=128,
        traffic_kwargs={},
    ),

    # More conservative PPO updates.
    # This may reduce the large drops seen in the PPO-LSTM learning curve.
    "stable": dict(
        learning_rate=1e-4,
        n_steps=1024,
        n_epochs=5,
        clip_range=0.1,
        ent_coef=0.003,
        lstm_hidden_size=128,
        traffic_kwargs={},
    ),

    # Larger LSTM memory.
    # This tests whether more memory helps the agent understand traffic history.
    "memory256": dict(
        learning_rate=1e-4,
        n_steps=1024,
        n_epochs=5,
        clip_range=0.1,
        ent_coef=0.003,
        lstm_hidden_size=256,
        traffic_kwargs={},
    ),

    # Longer rollout sequence.
    # This may help with boot delay and longer traffic patterns.
    "long_sequence": dict(
        learning_rate=1e-4,
        n_steps=2048,
        n_epochs=5,
        clip_range=0.1,
        ent_coef=0.003,
        lstm_hidden_size=256,
        traffic_kwargs={},
    ),

    # Trains on slightly harder bursty traffic.
    # This should be reported as a robustness variant, not as the original fair model.
    "robust_spikes": dict(
        learning_rate=1e-4,
        n_steps=2048,
        n_epochs=5,
        clip_range=0.1,
        ent_coef=0.005,
        lstm_hidden_size=256,
        traffic_kwargs={
            "spike_probability": 0.08,
            "spike_multiplier": 3.0,
        },
    ),
}


def plot_eval_curve(eval_path, out_path, label):
    """Create a plot showing how this PPO-LSTM variant improves during training."""

    if not os.path.exists(eval_path):
        print(f"No eval file found at {eval_path}")
        return

    data = np.load(eval_path)
    timesteps = data["timesteps"]
    rewards = data["results"]

    mean_rewards = rewards.mean(axis=1)
    std_rewards = rewards.std(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, mean_rewards, label=label)
    plt.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.25,
    )

    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation reward")
    plt.title(f"{label} Evaluation Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved plot to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO-LSTM variants on CloudScaling-v1")

    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--variant",
        default="stable",
        choices=list(LSTM_VARIANTS.keys()),
        help="PPO-LSTM variant to train.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = LSTM_VARIANTS[args.variant]
    run_name = f"recurrent_ppo_{args.variant}"
    label = f"PPO-LSTM {args.variant}"

    for path in [
        f"./models/best_{run_name}",
        f"./checkpoints/{run_name}",
        f"./logs/{run_name}",
        f"./logs/{run_name}_eval",
        f"./results/{run_name}",
    ]:
        os.makedirs(path, exist_ok=True)

    # Most variants use the original traffic.
    # The robust_spikes variant trains on harder spike traffic.
    env_kwargs = {}
    if cfg["traffic_kwargs"]:
        env_kwargs["traffic_kwargs"] = cfg["traffic_kwargs"]

    train_env = make_vec_env(
        n_envs=8,
        seed=args.seed,
        use_subprocess=False,
        norm_reward=True,
        **env_kwargs,
    )

    eval_env = VecNormalize(
        DummyVecEnv([make_env(rank=100, seed=args.seed, **env_kwargs)]),
        norm_obs=True,
        norm_reward=False,
        clip_obs=5.0,
        gamma=0.99,
    )
    eval_env.training = False

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=256,
        n_epochs=cfg["n_epochs"],
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            lstm_hidden_size=cfg["lstm_hidden_size"],
            n_lstm_layers=1,
            shared_lstm=False,
            enable_critic_lstm=True,
            net_arch=dict(pi=[256], vf=[256]),
        ),
        tensorboard_log=f"./logs/{run_name}/",
        device=args.device,
        seed=args.seed,
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/best_{run_name}/",
        log_path=f"./logs/{run_name}_eval/",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=f"./checkpoints/{run_name}/",
    )

    metrics_cb = MetricsCallback()

    print("=" * 60)
    print(f"[START] {label} Training")
    print(f"Device: {args.device} | Timesteps: {args.timesteps:,}")
    print(f"Variant config: {cfg}")
    print("=" * 60)

    start = time.perf_counter()

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[eval_cb, ckpt_cb, metrics_cb],
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Saving {label} model")

    model.save(f"./models/final_{run_name}")
    train_env.save(f"./models/vecnormalize_{run_name}.pkl")

    wall_time = time.perf_counter() - start

    plot_eval_curve(
        eval_path=f"./logs/{run_name}_eval/evaluations.npz",
        out_path=f"./results/{run_name}/{run_name}_eval_curve.png",
        label=label,
    )

    print("=" * 60)
    print(f"[DONE] {label} Training")
    print(f"Wall time: {wall_time:.1f}s")
    print(f"Final model: ./models/final_{run_name}.zip")
    print(f"Best model:  ./models/best_{run_name}/best_model.zip")
    print(f"VecNormalize: ./models/vecnormalize_{run_name}.pkl")
    print("=" * 60)

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()