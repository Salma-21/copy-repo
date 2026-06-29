"""
Train A2C variants on CloudScaling-v1.

here we keep the original A2C training structure, but adds a variant so we can test different A2C settings without creating many separate
files.

The goal is to improve A2C's behavior. In the previous results, A2C was very
safe: it had low queue occupancy and zero dropped requests. However, it used
many servers, so its cost was high. These variants mainly test whether A2C can
reduce over-provisioning while still protecting service quality.
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cloud_env import CloudScalingEnv  # noqa: F401
from env_factory import make_env, make_vec_env
from metrics_callback import MetricsCallback


A2C_VARIANTS = {
    # Similar to the original A2C setup.
    "balanced": dict(
        learning_rate=3e-4,
        n_steps=512,
        ent_coef=0.01,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    ),

    # Lower entropy means the policy should act less randomly.
    # A smaller network may also reduce over-aggressive behavior.
    "cost_aware": dict(
        learning_rate=5e-4,
        n_steps=256,
        ent_coef=0.003,
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    ),

    # This version gives A2C a longer rollout before each update.
    # It may help because scaling actions have delayed effects due to boot time.
    "long_rollout": dict(
        learning_rate=3e-4,
        n_steps=1024,
        ent_coef=0.005,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    ),

    # This version uses a smaller learning rate for smoother training.
    "low_lr_stable": dict(
        learning_rate=1e-4,
        n_steps=512,
        ent_coef=0.005,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    ),

    # This version keeps stronger exploration.
    # It may help if the policy becomes too fixed too early.
    "sla_safe": dict(
        learning_rate=3e-4,
        n_steps=512,
        ent_coef=0.02,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    ),
}


def plot_eval_curve(eval_path, out_path, label):
    """Create a plot showing how this A2C variant improves during training."""

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
    parser = argparse.ArgumentParser(description="Train A2C variants on CloudScaling-v1")

    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--variant",
        default="cost_aware",
        choices=list(A2C_VARIANTS.keys()),
        help="A2C variant to train.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = A2C_VARIANTS[args.variant]
    run_name = f"a2c_{args.variant}"
    label = f"A2C {args.variant}"

    for path in [
        f"./models/best_{run_name}",
        f"./checkpoints/{run_name}",
        f"./logs/{run_name}",
        f"./logs/{run_name}_eval",
        f"./results/{run_name}",
    ]:
        os.makedirs(path, exist_ok=True)

    train_env = make_vec_env(
        n_envs=8,
        seed=args.seed,
        use_subprocess=False,
        norm_reward=True,
    )

    eval_env = VecNormalize(
        DummyVecEnv([make_env(rank=100, seed=args.seed)]),
        norm_obs=True,
        norm_reward=False,
        clip_obs=5.0,
        gamma=0.99,
    )
    eval_env.training = False

    model = A2C(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=cfg["ent_coef"],
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=cfg["net_arch"]),
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