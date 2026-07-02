"""Traffic stress test for final trained autoscaling agents."""

import argparse
import csv
import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agent_adapters import BaselineAdapter, RecurrentPPOAdapter, SB3AgentAdapter
from baseline_agent import RuleBasedBaseline
from env_factory import make_env

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

try:
    from custom_dqn_policies import DuelingDQNPolicy, DoubleDuelingDQNPolicy
except ImportError:
    from dueling_dqn import DuelingDQNPolicy
    from double_dueling_dqn import DoubleDuelingDQNPolicy


@dataclass
class ModelSpec:
    name: str
    model_type: str
    model_path: str = None
    vecnorm_path: str = None
    model_class: object = None
    custom_policy: object = None


@dataclass
class TrafficSpec:
    name: str
    env_kwargs: dict


MODEL_SPECS = [
    ModelSpec("PPO", "sb3", "./models/best_ppo/best_model.zip", "./models/vecnormalize_ppo.pkl", PPO),
    ModelSpec("A2C Final", "sb3", "./models/best_final_a2c/best_model.zip", "./models/best_final_a2c/vecnormalize.pkl", A2C),
    ModelSpec("PPO-LSTM Final", "recurrent", "./models/best_final_recurrent_ppo/best_model.zip", "./models/best_final_recurrent_ppo/vecnormalize.pkl"),
    ModelSpec("Vanilla DQN freq1", "dqn", "./models/best_vanilla_dqn_freq1/best_model.zip", "./models/vecnormalize_vanilla_dqn_freq1.pkl"),
    ModelSpec("Double DQN freq1", "dqn", "./models/best_double_dqn_freq1/best_model.zip", "./models/vecnormalize_double_dqn_freq1.pkl"),
    ModelSpec("Dueling DQN freq1", "dqn", "./models/best_dueling_dqn_freq1/best_model.zip", "./models/vecnormalize_dueling_dqn_freq1.pkl", custom_policy=DuelingDQNPolicy),
    ModelSpec("Double+Dueling DQN freq1", "dqn", "./models/best_double_dueling_dqn_freq1/best_model.zip", "./models/vecnormalize_double_dueling_dqn_freq1.pkl", custom_policy=DoubleDuelingDQNPolicy),
    ModelSpec("Rule-Based Baseline", "baseline"),
]


TRAFFIC_CASES = [
    TrafficSpec("deterministic", {"traffic_mode": "deterministic", "traffic_kwargs": {"spike_probability": 0.0}}),
    TrafficSpec("poisson_only", {"traffic_mode": "stochastic", "traffic_kwargs": {"spike_probability": 0.0}}),
    TrafficSpec("bursty_spikes", {"traffic_mode": "stochastic", "traffic_kwargs": {"spike_probability": 0.05, "spike_multiplier": 3.0}}),
]


def parse_seeds(seed_text):
    return [int(seed.strip()) for seed in seed_text.split(",") if seed.strip()]


def make_eval_env(seed, vecnorm_path=None, env_kwargs=None):
    env_kwargs = env_kwargs or {}
    env = DummyVecEnv([make_env(rank=0, seed=seed, **env_kwargs)])

    if vecnorm_path is not None:
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    return env


def load_dqn_model(spec):
    if spec.custom_policy is None:
        return DQN.load(spec.model_path)

    return DQN.load(
        spec.model_path,
        custom_objects={"policy_class": spec.custom_policy},
    )


def load_adapter(spec):
    if spec.model_type == "baseline":
        baseline = RuleBasedBaseline()
        return BaselineAdapter(spec.name, baseline), None

    if not os.path.exists(spec.model_path):
        print(f"[SKIP] {spec.name}: missing model {spec.model_path}")
        return None, None

    if not os.path.exists(spec.vecnorm_path):
        print(f"[SKIP] {spec.name}: missing VecNormalize {spec.vecnorm_path}")
        return None, None

    if spec.model_type == "recurrent":
        if RecurrentPPO is None:
            print(f"[SKIP] {spec.name}: sb3-contrib is not installed.")
            return None, None
        model = RecurrentPPO.load(spec.model_path)
        return RecurrentPPOAdapter(spec.name, model), spec.vecnorm_path

    if spec.model_type == "dqn":
        model = load_dqn_model(spec)
        return SB3AgentAdapter(spec.name, model), spec.vecnorm_path

    model = spec.model_class.load(spec.model_path)
    return SB3AgentAdapter(spec.name, model), spec.vecnorm_path


def run_episode(adapter, env, max_queue=500):
    obs = env.reset()
    done = np.array([False])
    adapter.reset_episode(num_envs=env.num_envs)

    total_return = 0.0
    total_cost = 0.0
    total_dropped = 0.0
    total_queue = 0.0
    previous_action = None
    action_switches = 0
    steps = 0

    while not done[0]:
        action = adapter.predict(obs, done)
        current_action = int(np.asarray(action).flatten()[0])

        if previous_action is not None and current_action != previous_action:
            action_switches += 1
        previous_action = current_action

        obs, reward, done, infos = env.step(action)
        info = infos[0]

        active_servers = info.get("active", info.get("active_servers", 0.0))
        dropped = info.get("dropped", info.get("dropped_requests", 0.0))
        queue = info.get("queue", info.get("queue_length", 0.0))

        total_return += float(reward[0])
        total_cost += float(active_servers)
        total_dropped += float(dropped)
        total_queue += float(queue)
        steps += 1

    mean_queue = total_queue / max(1, steps)
    switch_rate = action_switches / max(1, steps - 1)

    return {
        "return": total_return,
        "latency_proxy": mean_queue / max_queue,
        "mean_queue": mean_queue,
        "cost": total_cost,
        "dropped_requests": total_dropped,
        "action_stability": 1.0 - switch_rate,
        "action_switches": action_switches,
        "steps": steps,
    }


def summarize(rows):
    metrics = [
        "return",
        "latency_proxy",
        "mean_queue",
        "cost",
        "dropped_requests",
        "action_stability",
        "action_switches",
    ]

    summary = {}
    for metric in metrics:
        values = np.array([row[metric] for row in rows], dtype=float)
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0

    return summary


def write_csv(path, rows):
    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def plot_grouped_metric(summary_rows, metric, out_dir):
    traffic_names = [case.name for case in TRAFFIC_CASES]
    algorithms = sorted({row["algorithm"] for row in summary_rows})
    x = np.arange(len(traffic_names))
    width = 0.8 / max(1, len(algorithms))

    plt.figure(figsize=(16, 6))

    for i, algorithm in enumerate(algorithms):
        means = []
        stds = []

        for traffic_name in traffic_names:
            match = [
                row for row in summary_rows
                if row["algorithm"] == algorithm and row["traffic_case"] == traffic_name
            ]
            means.append(match[0][f"{metric}_mean"] if match else 0.0)
            stds.append(match[0][f"{metric}_std"] if match else 0.0)

        offset = (i - (len(algorithms) - 1) / 2) * width
        plt.bar(x + offset, means, width, yerr=stds, capsize=3, label=algorithm)

    plt.xticks(x, traffic_names)
    plt.ylabel(metric.replace("_", " "))
    plt.title(f"Traffic Stress Test: {metric.replace('_', ' ')}")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f"{metric}.png"), dpi=200)
    plt.close()


def plot_traffic_curve(summary_rows, metric, out_dir):
    traffic_names = [case.name for case in TRAFFIC_CASES]
    algorithms = sorted({row["algorithm"] for row in summary_rows})

    plt.figure(figsize=(12, 5))

    for algorithm in algorithms:
        values = []

        for traffic_name in traffic_names:
            match = [
                row for row in summary_rows
                if row["algorithm"] == algorithm and row["traffic_case"] == traffic_name
            ]
            values.append(match[0][f"{metric}_mean"] if match else np.nan)

        plt.plot(traffic_names, values, marker="o", linewidth=2, label=algorithm)

    plt.ylabel(metric.replace("_", " "))
    plt.title(f"Traffic Stress Curve: {metric.replace('_', ' ')}")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f"{metric}_traffic_curve.png"), dpi=200)
    plt.close()


def plot_all_traffic_curves(summary_rows, out_dir):
    metrics = [
        "return",
        "latency_proxy",
        "cost",
        "dropped_requests",
        "action_stability",
    ]

    traffic_names = [case.name for case in TRAFFIC_CASES]
    algorithms = sorted({row["algorithm"] for row in summary_rows})

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        for algorithm in algorithms:
            values = []

            for traffic_name in traffic_names:
                match = [
                    row for row in summary_rows
                    if row["algorithm"] == algorithm and row["traffic_case"] == traffic_name
                ]
                values.append(match[0][f"{metric}_mean"] if match else np.nan)

            ax.plot(traffic_names, values, marker="o", linewidth=2, label=algorithm)

        ax.set_title(metric.replace("_", " "))
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=15)

    axes[-1].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("Traffic Stress Test Summary", fontsize=14)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))

    fig.savefig(os.path.join(out_dir, "all_traffic_curves.png"), dpi=200)
    plt.close(fig)


def print_summary(summary_rows):
    print("\n=== Traffic Stress Test Summary ===")

    for traffic_case in [case.name for case in TRAFFIC_CASES]:
        print(f"\nTraffic case: {traffic_case}")

        rows = [row for row in summary_rows if row["traffic_case"] == traffic_case]
        for row in rows:
            print(f"  {row['algorithm']}")
            print(f"    Return: {row['return_mean']:.2f} +/- {row['return_std']:.2f}")
            print(f"    Latency proxy: {row['latency_proxy_mean']:.4f} +/- {row['latency_proxy_std']:.4f}")
            print(f"    Cost: {row['cost_mean']:.2f} +/- {row['cost_std']:.2f}")
            print(f"    Dropped requests: {row['dropped_requests_mean']:.2f} +/- {row['dropped_requests_std']:.2f}")
            print(f"    Action stability: {row['action_stability_mean']:.4f} +/- {row['action_stability_std']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate final models under deterministic, Poisson, and bursty traffic."
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated evaluation seeds.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="./results/traffic_stress_test",
        help="Folder where CSV, JSON, and plots will be saved.",
    )

    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)

    adapters_with_vecnorm = []
    for spec in MODEL_SPECS:
        adapter, vecnorm_path = load_adapter(spec)
        if adapter is not None:
            adapters_with_vecnorm.append((adapter, vecnorm_path))

    if not adapters_with_vecnorm:
        raise RuntimeError("No models were loaded. Check model paths.")

    all_episode_rows = []
    summary_rows = []

    for traffic_case in TRAFFIC_CASES:
        print(f"\n=== Evaluating traffic case: {traffic_case.name} ===")

        for adapter, vecnorm_path in adapters_with_vecnorm:
            print(f"Evaluating {adapter.name}...")

            rows_for_group = []

            for seed in seeds:
                env = make_eval_env(
                    seed=seed,
                    vecnorm_path=vecnorm_path,
                    env_kwargs=traffic_case.env_kwargs,
                )

                metrics = run_episode(adapter, env)
                env.close()

                row = {
                    "traffic_case": traffic_case.name,
                    "algorithm": adapter.name,
                    "seed": seed,
                    **metrics,
                }

                all_episode_rows.append(row)
                rows_for_group.append(row)

            summary = summarize(rows_for_group)
            summary_rows.append({
                "traffic_case": traffic_case.name,
                "algorithm": adapter.name,
                "num_seeds": len(seeds),
                **summary,
            })

    os.makedirs(args.out_dir, exist_ok=True)

    write_csv(os.path.join(args.out_dir, "episode_results.csv"), all_episode_rows)
    write_csv(os.path.join(args.out_dir, "summary_results.csv"), summary_rows)
    write_json(os.path.join(args.out_dir, "summary_results.json"), summary_rows)

    for metric in [
        "return",
        "latency_proxy",
        "cost",
        "dropped_requests",
        "action_stability",
    ]:
        plot_grouped_metric(summary_rows, metric, args.out_dir)
        plot_traffic_curve(summary_rows, metric, args.out_dir)

    plot_all_traffic_curves(summary_rows, args.out_dir)

    print_summary(summary_rows)
    print(f"\nSaved results to: {args.out_dir}")


if __name__ == "__main__":
    main()
