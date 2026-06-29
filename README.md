# copy-repo

###################################################

import argparse
import json
import os

import numpy as np
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_factory import make_env
from baseline_agent import RuleBasedBaseline

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None


def evaluate_agent(
    model,
    vecnorm_path=None,
    n_episodes=10,
    seed=42,
    is_recurrent=False,
):
    """Evaluate a policy deterministically.

    Returns mean/std for:
    - total reward
    - operational cost
    - dropped requests
    - queue occupancy rate
    """

    env = DummyVecEnv([make_env(rank=100, seed=seed)])

    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    rewards, costs, drops, qocc = [], [], [], []

    for _ in range(n_episodes):
        obs = env.reset()
        done = np.array([False])

        total_reward = 0.0
        total_cost = 0.0
        total_dropped = 0.0
        total_queue = 0.0
        steps = 0

        lstm_state = None
        episode_start = np.ones((env.num_envs,), dtype=bool)

        while not done[0]:
            if model == "random":
                action = np.array([env.action_space.sample()])

            elif is_recurrent:
                action, lstm_state = model.predict(
                    obs,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=True,
                )

            elif hasattr(model, "predict"):
                action, _ = model.predict(obs, deterministic=True)

            else:
                action = np.array([1])

            obs, reward, done, info = env.step(action)
            episode_start = done

            total_reward += float(reward[0])
            total_cost += float(info[0]["active"])
            total_dropped += float(info[0]["dropped"])
            total_queue += float(info[0]["queue"])
            steps += 1

        rewards.append(total_reward)
        costs.append(total_cost)
        drops.append(total_dropped)
        qocc.append(total_queue / (steps * 500))

    env.close()

    def agg(values):
        return float(np.mean(values)), float(np.std(values))

    return {
        "reward": agg(rewards),
        "cost": agg(costs),
        "dropped": agg(drops),
        "queue_occ": agg(qocc),
    }


def maybe_eval_model(
    results,
    name,
    model_class,
    model_path,
    vecnorm_path,
    episodes,
    seed,
    is_recurrent=False,
):
    print(f"Evaluating {name}...")

    if not os.path.exists(model_path):
        print(f"{name} model not found. Skipping.")
        return

    if not os.path.exists(vecnorm_path):
        print(f"{name} VecNormalize file not found. Skipping.")
        return

    model = model_class.load(model_path)

    results[name] = evaluate_agent(
        model,
        vecnorm_path=vecnorm_path,
        n_episodes=episodes,
        seed=seed,
        is_recurrent=is_recurrent,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained policies deterministically"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="./results/eval_summary.json",
        help="Path to save evaluation summary",
    )

    args = parser.parse_args()

    results = {}

    print("Evaluating Baseline...")
    baseline_model = RuleBasedBaseline()
    results["Baseline"] = evaluate_agent(
        baseline_model,
        vecnorm_path=None,
        n_episodes=args.episodes,
        seed=args.seed,
    )

    print("Evaluating Random...")
    results["Random"] = evaluate_agent(
        "random",
        vecnorm_path=None,
        n_episodes=args.episodes,
        seed=args.seed,
    )

    maybe_eval_model(
        results,
        name="PPO",
        model_class=PPO,
        model_path="./models/best_ppo/best_model.zip",
        vecnorm_path="./models/vecnormalize_ppo.pkl",
        episodes=args.episodes,
        seed=args.seed,
    )

    maybe_eval_model(
        results,
        name="DQN",
        model_class=DQN,
        model_path="./models/best_dqn/best_model.zip",
        vecnorm_path="./models/vecnormalize_dqn.pkl",
        episodes=args.episodes,
        seed=args.seed,
    )

    maybe_eval_model(
        results,
        name="A2C",
        model_class=A2C,
        model_path="./models/best_a2c/best_model.zip",
        vecnorm_path="./models/vecnormalize_a2c.pkl",
        episodes=args.episodes,
        seed=args.seed,
    )

    if RecurrentPPO is not None:
        maybe_eval_model(
            results,
            name="PPO-LSTM",
            model_class=RecurrentPPO,
            model_path="./models/best_recurrent_ppo/best_model.zip",
            vecnorm_path="./models/vecnormalize_recurrent_ppo.pkl",
            episodes=args.episodes,
            seed=args.seed,
            is_recurrent=True,
        )
    else:
        print("sb3-contrib not installed. Skipping PPO-LSTM.")

    print("\n--- Final Evaluation Results ---")

    for agent, metrics in results.items():
        print(f"\n{agent}:")
        print(f"  Reward:      {metrics['reward'][0]:.2f} +/- {metrics['reward'][1]:.2f}")
        print(f"  Cost:        {metrics['cost'][0]:.2f} +/- {metrics['cost'][1]:.2f}")
        print(f"  Dropped:     {metrics['dropped'][0]:.2f} +/- {metrics['dropped'][1]:.2f}")
        print(f"  Queue Occ:   {metrics['queue_occ'][0]:.4f} +/- {metrics['queue_occ'][1]:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved evaluation summary to: {args.out}")


if __name__ == "__main__":
    main()
