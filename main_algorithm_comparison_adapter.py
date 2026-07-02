"""Main algorithm comparison for the cloud autoscaling experiments.

This script evaluates the selected final models under the same environment,
traffic generator, reward function, action space, and random seeds. It is used
to produce the main algorithm comparison table and plots in the report.

Models included:
- PPO
- A2C Final
- PPO-LSTM Final
- Vanilla DQN with update_frequency=1
- Double DQN with update_frequency=1
- Dueling DQN with update_frequency=1
- Double+Dueling DQN with update_frequency=1
- Rule-Based Baseline

We only loads already trained models and evaluates them fairly using the same evaluation seeds.

Metrics reported:
- cumulative return
- latency proxy
- mean queue length
- cost
- dropped requests
- action stability
"""

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
    """Configuration for one evaluated policy.

    model_type controls how the policy is loaded:
    - "sb3" for standard Stable-Baselines3 models such as PPO and A2C
    - "dqn" for DQN variants, including custom dueling policies
    - "recurrent" for PPO-LSTM / RecurrentPPO
    - "baseline" for the threshold-based rule baseline

    The baseline does not need model_path or vecnorm_path because it is not a
    trained neural policy.
    """

    name: str
    model_type: str
    model_path: str = None
    vecnorm_path: str = None
    model_class: object = None
    custom_policy: object = None


MODEL_SPECS = [
    ModelSpec(
        name="PPO",
        model_type="sb3",
        model_path="./models/best_ppo/best_model.zip",
        vecnorm_path="./models/vecnormalize_ppo.pkl",
        model_class=PPO,
    ),
    ModelSpec(
        name="A2C Final",
        model_type="sb3",
        model_path="./models/best_final_a2c/best_model.zip",
        vecnorm_path="./models/best_final_a2c/vecnormalize.pkl",
        model_class=A2C,
    ),
    ModelSpec(
        name="PPO-LSTM Final",
        model_type="recurrent",
        model_path="./models/best_final_recurrent_ppo/best_model.zip",
        vecnorm_path="./models/best_final_recurrent_ppo/vecnormalize.pkl",
    ),
    ModelSpec(
        name="Vanilla DQN freq1",
        model_type="dqn",
        model_path="./models/best_vanilla_dqn_freq1/best_model.zip",
        vecnorm_path="./models/vecnormalize_vanilla_dqn_freq1.pkl",
    ),
    ModelSpec(
        name="Double DQN freq1",
        model_type="dqn",
        model_path="./models/best_double_dqn_freq1/best_model.zip",
        vecnorm_path="./models/vecnormalize_double_dqn_freq1.pkl",
    ),
    ModelSpec(
        name="Dueling DQN freq1",
        model_type="dqn",
        model_path="./models/best_dueling_dqn_freq1/best_model.zip",
        vecnorm_path="./models/vecnormalize_dueling_dqn_freq1.pkl",
        custom_policy=DuelingDQNPolicy,
    ),
    ModelSpec(
        name="Double+Dueling DQN freq1",
        model_type="dqn",
        model_path="./models/best_double_dueling_dqn_freq1/best_model.zip",
        vecnorm_path="./models/vecnormalize_double_dueling_dqn_freq1.pkl",
        custom_policy=DoubleDuelingDQNPolicy,
    ),
    ModelSpec(
        name="Rule-Based Baseline",
        model_type="baseline",
    ),
]


def parse_seeds(seed_text):
    """Convert a comma-separated seed string into a list of integers."""

    return [int(seed.strip()) for seed in seed_text.split(",") if seed.strip()]


def make_eval_env(seed, vecnorm_path=None):
    """Create one normalized evaluation environment.

    Each model is evaluated using the same environment definition and seed. If
    the model was trained with VecNormalize, the saved normalization statistics
    are loaded and frozen for evaluation.

    The rule-based baseline does not use VecNormalize, so vecnorm_path can be
    None.
    """

    env = DummyVecEnv([make_env(rank=0, seed=seed)])

    if vecnorm_path is not None:
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    return env


def load_dqn_model(spec):
    """Load a DQN model.

    Standard Vanilla DQN and Double DQN can be loaded directly. Dueling DQN and
    Double+Dueling DQN need their custom policy classes so Stable-Baselines3 can
    reconstruct the saved model correctly.
    """

    if spec.custom_policy is None:
        return DQN.load(spec.model_path)

    return DQN.load(
        spec.model_path,
        custom_objects={"policy_class": spec.custom_policy},
    )


def load_adapter(spec):
    """Load one model and wrap it with a common adapter interface.

    Different algorithms expose slightly different prediction APIs. The adapter
    layer hides those differences so the evaluation loop can call predict() in
    the same way for PPO, A2C, PPO-LSTM, DQN variants, and the rule-based
    baseline.
    """

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
    """Run one evaluation episode and collect autoscaling metrics.

    The returned metrics are the values used in the final comparison table:
    return, latency proxy, cost, dropped requests, and action stability.

    The environment does not directly report request-level latency, so the
    latency proxy is computed from the average queue length normalized by the
    maximum queue size.
    """

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
    """Compute mean and standard deviation over the evaluation seeds."""

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
        summary[f"{metric}_std"] = (
            float(values.std(ddof=1)) if len(values) > 1 else 0.0
        )

    return summary


def write_csv(path, rows):
    """Save result rows to a CSV file."""

    if not rows:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    """Save summary data as a JSON file for easier inspection."""

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def plot_metric(summary_rows, metric, out_dir):
    """Create a bar plot with error bars for one metric.

    The bars show the mean over seeds, and the error bars show the standard
    deviation. These plots are useful for seeing both performance and
    consistency.
    """

    names = [row["algorithm"] for row in summary_rows]
    means = [row[f"{metric}_mean"] for row in summary_rows]
    stds = [row[f"{metric}_std"] for row in summary_rows]

    plt.figure(figsize=(14, 5))
    plt.bar(names, means, yerr=stds, capsize=4)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric.replace("_", " "))
    plt.title(f"Main Algorithm Comparison: {metric.replace('_', ' ')}")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f"{metric}.png"), dpi=200)
    plt.close()


def plot_metric_by_seed(episode_rows, metric, out_dir):
    """Plot each algorithm's metric value across the individual seeds.

    This helps identify whether an algorithm is consistently strong or only
    performs well on some traffic seeds.
    """

    algorithms = sorted({row["algorithm"] for row in episode_rows})

    plt.figure(figsize=(12, 5))

    for algorithm in algorithms:
        rows = sorted(
            [row for row in episode_rows if row["algorithm"] == algorithm],
            key=lambda row: row["seed"],
        )

        seeds = [row["seed"] for row in rows]
        values = [row[metric] for row in rows]

        plt.plot(seeds, values, marker="o", linewidth=2, label=algorithm)

    plt.xlabel("Evaluation seed")
    plt.ylabel(metric.replace("_", " "))
    plt.title(f"Per-seed comparison: {metric.replace('_', ' ')}")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, f"{metric}_by_seed.png"), dpi=200)
    plt.close()


def plot_combined_metrics(summary_rows, out_dir):
    """Create one summary figure containing all main comparison metrics."""

    metrics = [
        "return",
        "latency_proxy",
        "cost",
        "dropped_requests",
        "action_stability",
    ]

    names = [row["algorithm"] for row in summary_rows]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        means = [row[f"{metric}_mean"] for row in summary_rows]
        stds = [row[f"{metric}_std"] for row in summary_rows]

        ax.bar(names, means, yerr=stds, capsize=4)
        ax.set_title(metric.replace("_", " "))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].axis("off")
    fig.suptitle("Main Algorithm Comparison Summary", fontsize=14)
    fig.tight_layout()

    fig.savefig(os.path.join(out_dir, "all_metrics_summary.png"), dpi=200)
    plt.close(fig)


def print_summary(summary_rows):
    """Print the same summary values that are saved to summary_results.csv."""

    print("\n=== Main Algorithm Comparison ===")

    for row in summary_rows:
        print(f"\n{row['algorithm']}")
        print(f"  Return:           {row['return_mean']:.2f} +/- {row['return_std']:.2f}")
        print(
            "  Latency proxy:    "
            f"{row['latency_proxy_mean']:.4f} +/- "
            f"{row['latency_proxy_std']:.4f}"
        )
        print(f"  Cost:             {row['cost_mean']:.2f} +/- {row['cost_std']:.2f}")
        print(
            "  Dropped requests: "
            f"{row['dropped_requests_mean']:.2f} +/- "
            f"{row['dropped_requests_std']:.2f}"
        )
        print(
            "  Action stability: "
            f"{row['action_stability_mean']:.4f} +/- "
            f"{row['action_stability_std']:.4f}"
        )


def main():
    """Run the full main algorithm comparison experiment.

    The script loads all available models, evaluates each one over the selected
    seeds, writes CSV/JSON result files, and saves the plots in the output
    folder.
    """

    parser = argparse.ArgumentParser(
        description="Evaluate final/tuned algorithms on the same traffic seeds."
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
        default="./results/main_algorithm_comparison",
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
        raise RuntimeError("No models were loaded. Check model and VecNormalize paths.")

    all_episode_rows = []
    summary_rows = []

    for adapter, vecnorm_path in adapters_with_vecnorm:
        print(f"\nEvaluating {adapter.name}...")

        rows_for_agent = []

        for seed in seeds:
            env = make_eval_env(seed=seed, vecnorm_path=vecnorm_path)
            metrics = run_episode(adapter, env)
            env.close()

            row = {
                "algorithm": adapter.name,
                "seed": seed,
                **metrics,
            }

            all_episode_rows.append(row)
            rows_for_agent.append(row)

        summary = summarize(rows_for_agent)
        summary_rows.append({
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
        plot_metric(summary_rows, metric, args.out_dir)
        plot_metric_by_seed(all_episode_rows, metric, args.out_dir)

    plot_combined_metrics(summary_rows, args.out_dir)

    print_summary(summary_rows)
    print(f"\nSaved results to: {args.out_dir}")


if __name__ == "__main__":
    main()
