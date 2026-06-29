"""
Traffic-stress test using the Adapter Design Pattern.

here we evaluates the trained agents under three traffic conditions:

1. Deterministic traffic
   Smooth sinusoidal traffic only.
   No Poisson randomness and no spikes.

2. Poisson-only traffic
   Smooth sinusoidal traffic with Poisson arrival randomness.
   No random spikes.

3. Bursty spike traffic
   Poisson arrivals plus random traffic spikes.

A good cloud autoscaling policy should not only work on one smooth workload.
It should also handle randomness and sudden bursts. This test keeps the trained
policy fixed and changes only the workload pattern.

The adapter pattern is used so all agents can be evaluated using the same loop.
This is important because PPO-LSTM needs recurrent memory, while normal SB3
models and the rule-based baseline do not.
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# These imports make custom DQN policy classes available when loading models.
import dueling_dqn  # noqa: F401
import double_dueling_dqn  # noqa: F401

from agent_adapters import SB3AgentAdapter, RecurrentPPOAdapter, BaselineAdapter
from baseline_agent import RuleBasedBaseline
from env_factory import make_env
from sparse_ppo import SparsePPO

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None


@dataclass
class ModelSpec:
    name: str
    model_class: object
    model_path: str
    vecnorm_path: str


MODEL_SPECS = [
    ModelSpec(
        "Vanilla DQN",
        DQN,
        "./models/best_vanilla_dqn_freq4/best_model.zip",
        "./models/vecnormalize_vanilla_dqn_freq4.pkl",
    ),
    ModelSpec(
        "Double DQN",
        DQN,
        "./models/best_double_dqn_freq4/best_model.zip",
        "./models/vecnormalize_double_dqn_freq4.pkl",
    ),
    ModelSpec(
        "Dueling DQN",
        DQN,
        "./models/best_dueling_dqn_freq4/best_model.zip",
        "./models/vecnormalize_dueling_dqn_freq4.pkl",
    ),
    ModelSpec(
        "Double-Dueling DQN",
        DQN,
        "./models/best_double_dueling_dqn_freq4/best_model.zip",
        "./models/vecnormalize_double_dueling_dqn_freq4.pkl",
    ),
    ModelSpec(
        "PPO",
        PPO,
        "./models/best_ppo/best_model.zip",
        "./models/vecnormalize_ppo.pkl",
    ),
    ModelSpec(
        "Sparse PPO K=4",
        SparsePPO,
        "./models/sparse_ppo_k4.zip",
        "./models/vecnormalize_sparse_ppo_k4.pkl",
    ),
    ModelSpec(
        "A2C",
        A2C,
        "./models/best_a2c/best_model.zip",
        "./models/vecnormalize_a2c.pkl",
    ),
]


TRAFFIC_CASES = {
    "deterministic": {
        "traffic_mode": "deterministic",
        "traffic_kwargs": {
            "spike_probability": 0.0,
        },
    },
    "poisson_only": {
        "traffic_mode": "stochastic",
        "traffic_kwargs": {
            "spike_probability": 0.0,
        },
    },
    "bursty_spikes": {
        "traffic_mode": "stochastic",
        "traffic_kwargs": {
            "spike_probability": 0.05,
            "spike_multiplier": 3.0,
        },
    },
}


def parse_seeds(seed_text):
    return [int(seed.strip()) for seed in seed_text.split(",") if seed.strip()]


def make_eval_env(seed, vecnorm_path, env_kwargs):
    """Create one evaluation environment for one traffic case."""

    env = DummyVecEnv([make_env(rank=0, seed=seed, **env_kwargs)])

    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    return env


def load_adapter(spec):
    """Load a trained non-recurrent model and wrap it with an adapter."""

    if not os.path.exists(spec.model_path):
        print(f"[SKIP] {spec.name}: missing model {spec.model_path}")
        return None

    if not os.path.exists(spec.vecnorm_path):
        print(f"[SKIP] {spec.name}: missing VecNormalize {spec.vecnorm_path}")
        return None

    model = spec.model_class.load(spec.model_path)
    return SB3AgentAdapter(spec.name, model)


def load_recurrent_ppo_adapter():
    """Load PPO-LSTM and wrap it with the recurrent adapter."""

    if RecurrentPPO is None:
        print("[SKIP] PPO-LSTM: sb3-contrib is not installed.")
        return None, None

    model_path = "./models/best_recurrent_ppo/best_model.zip"
    vecnorm_path = "./models/vecnormalize_recurrent_ppo.pkl"

    if not os.path.exists(model_path):
        print(f"[SKIP] PPO-LSTM: missing model {model_path}")
        return None, None

    if not os.path.exists(vecnorm_path):
        print(f"[SKIP] PPO-LSTM: missing VecNormalize {vecnorm_path}")
        return None, None

    model = RecurrentPPO.load(model_path)
    return RecurrentPPOAdapter("PPO-LSTM", model), vecnorm_path


def run_episode(adapter, env, max_queue=500):
    """Run one evaluation episode and collect cloud metrics."""

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

        total_return += float(reward[0])
        total_cost += float(info["active"])
        total_dropped += float(info["dropped"])
        total_queue += float(info["queue"])

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
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def plot_grouped_metric(summary_rows, metric, out_dir):
    """Plot one metric for all algorithms under all traffic cases."""

    algorithms = [row["algorithm"] for row in summary_rows]
    algorithms = list(dict.fromkeys(algorithms))

    traffic_cases = list(TRAFFIC_CASES.keys())

    lookup = {
        (row["algorithm"], row["traffic_case"]): row
        for row in summary_rows
    }

    x = np.arange(len(algorithms))
    width = 0.25

    plt.figure(figsize=(14, 6))

    for i, traffic_case in enumerate(traffic_cases):
        means = []
        stds = []

        for algorithm in algorithms:
            row = lookup.get((algorithm, traffic_case))
            if row is None:
                means.append(0.0)
                stds.append(0.0)
            else:
                means.append(row[f"{metric}_mean"])
                stds.append(row[f"{metric}_std"])

        plt.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=traffic_case,
        )

    if metric == "return":
        ylabel = "return"
        note = "closer to zero is better"
    elif metric == "latency_proxy":
        ylabel = "latency proxy"
        note = "lower is better"
    elif metric in {"cost", "dropped_requests"}:
        ylabel = metric.replace("_", " ")
        note = "lower is better"
    else:
        ylabel = metric.replace("_", " ")
        note = "higher is better"

    plt.xticks(x, algorithms, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(f"Traffic Stress Test: {ylabel} ({note})")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, f"{metric}.png")
    plt.savefig(path, dpi=200)
    plt.close()


def print_summary(summary_rows):
    print("\n=== Traffic Stress Test Summary ===")

    for row in summary_rows:
        print(f"\n{row['algorithm']} | {row['traffic_case']}")
        print(f"  Return:           {row['return_mean']:.2f} +/- {row['return_std']:.2f}")
        print(f"  Latency proxy:    {row['latency_proxy_mean']:.4f} +/- {row['latency_proxy_std']:.4f}")
        print(f"  Cost:             {row['cost_mean']:.2f} +/- {row['cost_std']:.2f}")
        print(f"  Dropped requests: {row['dropped_requests_mean']:.2f} +/- {row['dropped_requests_std']:.2f}")
        print(f"  Action stability: {row['action_stability_mean']:.4f} +/- {row['action_stability_std']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Traffic stress test with adapters")

    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated evaluation traffic seeds.",
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

    baseline_adapter = BaselineAdapter(RuleBasedBaseline())
    adapters_with_vecnorm.append((baseline_adapter, None))

    for spec in MODEL_SPECS:
        adapter = load_adapter(spec)
        if adapter is not None:
            adapters_with_vecnorm.append((adapter, spec.vecnorm_path))

    recurrent_adapter, recurrent_vecnorm = load_recurrent_ppo_adapter()
    if recurrent_adapter is not None:
        adapters_with_vecnorm.append((recurrent_adapter, recurrent_vecnorm))

    all_episode_rows = []
    summary_rows = []

    for adapter, vecnorm_path in adapters_with_vecnorm:
        for traffic_case, env_kwargs in TRAFFIC_CASES.items():
            print(f"\nEvaluating {adapter.name} on {traffic_case} traffic...")

            case_rows = []

            for seed in seeds:
                env = make_eval_env(
                    seed=seed,
                    vecnorm_path=vecnorm_path,
                    env_kwargs=env_kwargs,
                )

                metrics = run_episode(adapter, env)
                env.close()

                row = {
                    "algorithm": adapter.name,
                    "traffic_case": traffic_case,
                    "seed": seed,
                    **metrics,
                }

                all_episode_rows.append(row)
                case_rows.append(row)

            summary = summarize(case_rows)
            summary["algorithm"] = adapter.name
            summary["traffic_case"] = traffic_case
            summary["num_seeds"] = len(seeds)

            summary_rows.append(summary)

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

    print_summary(summary_rows)
    print(f"\nSaved traffic stress-test results to: {args.out_dir}")


if __name__ == "__main__":
    main()