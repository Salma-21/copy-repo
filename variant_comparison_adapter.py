"""
Compare original A2C/PPO-LSTM models with their tuned variants.

This file loads already-trained models and evaluates them on the same
traffic seeds. It does not train anything.

It selects:
- the best A2C model from A2C Original + A2C variants
- the best PPO-LSTM model from PPO-LSTM Original + PPO-LSTM variants

The selected final models are saved into:
- ./models/best_final_a2c/
- ./models/best_final_recurrent_ppo/

The reward is a negative penalty, so return values are usually negative.
A return closer to zero is better.
"""

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agent_adapters import SB3AgentAdapter, RecurrentPPOAdapter
from env_factory import make_env

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None


@dataclass
class VariantSpec:
    name: str
    family: str
    model_class: object
    model_paths: list
    vecnorm_paths: list
    recurrent: bool = False


VARIANT_SPECS = [
    VariantSpec(
        "A2C Original",
        "A2C",
        A2C,
        ["./models/best_a2c/best_model.zip", "./models/final_a2c.zip"],
        ["./models/vecnormalize_a2c.pkl"],
    ),
    VariantSpec(
        "A2C Balanced",
        "A2C",
        A2C,
        ["./models/best_a2c_balanced/best_model.zip", "./models/final_a2c_balanced.zip"],
        ["./models/vecnormalize_a2c_balanced.pkl"],
    ),
    VariantSpec(
        "A2C Cost-Aware",
        "A2C",
        A2C,
        ["./models/best_a2c_cost_aware/best_model.zip", "./models/final_a2c_cost_aware.zip"],
        ["./models/vecnormalize_a2c_cost_aware.pkl"],
    ),
    VariantSpec(
        "A2C Long-Rollout",
        "A2C",
        A2C,
        ["./models/best_a2c_long_rollout/best_model.zip", "./models/final_a2c_long_rollout.zip"],
        ["./models/vecnormalize_a2c_long_rollout.pkl"],
    ),
    VariantSpec(
        "A2C Low-LR-Stable",
        "A2C",
        A2C,
        ["./models/best_a2c_low_lr_stable/best_model.zip", "./models/final_a2c_low_lr_stable.zip"],
        ["./models/vecnormalize_a2c_low_lr_stable.pkl"],
    ),
    VariantSpec(
        "A2C SLA-Safe",
        "A2C",
        A2C,
        ["./models/best_a2c_sla_safe/best_model.zip", "./models/final_a2c_sla_safe.zip"],
        ["./models/vecnormalize_a2c_sla_safe.pkl"],
    ),
    VariantSpec(
        "PPO-LSTM Original",
        "PPO-LSTM",
        None,
        ["./models/best_recurrent_ppo/best_model.zip", "./models/final_recurrent_ppo.zip"],
        ["./models/vecnormalize_recurrent_ppo.pkl"],
        recurrent=True,
    ),
    VariantSpec(
        "PPO-LSTM Balanced",
        "PPO-LSTM",
        None,
        [
            "./models/best_recurrent_ppo_balanced/best_model.zip",
            "./models/final_recurrent_ppo_balanced.zip",
        ],
        ["./models/vecnormalize_recurrent_ppo_balanced.pkl"],
        recurrent=True,
    ),
    VariantSpec(
        "PPO-LSTM Stable",
        "PPO-LSTM",
        None,
        [
            "./models/best_recurrent_ppo_stable/best_model.zip",
            "./models/final_recurrent_ppo_stable.zip",
        ],
        ["./models/vecnormalize_recurrent_ppo_stable.pkl"],
        recurrent=True,
    ),
    VariantSpec(
        "PPO-LSTM Memory256",
        "PPO-LSTM",
        None,
        [
            "./models/best_recurrent_ppo_memory256/best_model.zip",
            "./models/final_recurrent_ppo_memory256.zip",
        ],
        ["./models/vecnormalize_recurrent_ppo_memory256.pkl"],
        recurrent=True,
    ),
    VariantSpec(
        "PPO-LSTM Long-Sequence",
        "PPO-LSTM",
        None,
        [
            "./models/best_recurrent_ppo_long_sequence/best_model.zip",
            "./models/final_recurrent_ppo_long_sequence.zip",
        ],
        ["./models/vecnormalize_recurrent_ppo_long_sequence.pkl"],
        recurrent=True,
    ),
    VariantSpec(
        "PPO-LSTM Robust-Spikes",
        "PPO-LSTM",
        None,
        [
            "./models/best_recurrent_ppo_robust_spikes/best_model.zip",
            "./models/final_recurrent_ppo_robust_spikes.zip",
        ],
        ["./models/vecnormalize_recurrent_ppo_robust_spikes.pkl"],
        recurrent=True,
    ),
]


def parse_seeds(seed_text):
    return [int(seed.strip()) for seed in seed_text.split(",") if seed.strip()]


def first_existing_path(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def get_env_kwargs(traffic_profile):
    if traffic_profile == "standard":
        return {}

    if traffic_profile == "deterministic":
        return {
            "traffic_mode": "deterministic",
            "traffic_kwargs": {"spike_probability": 0.0},
        }

    if traffic_profile == "poisson_only":
        return {
            "traffic_mode": "stochastic",
            "traffic_kwargs": {"spike_probability": 0.0},
        }

    if traffic_profile == "bursty_spikes":
        return {
            "traffic_mode": "stochastic",
            "traffic_kwargs": {
                "spike_probability": 0.05,
                "spike_multiplier": 3.0,
            },
        }

    raise ValueError(f"Unknown traffic profile: {traffic_profile}")


def make_eval_env(seed, vecnorm_path, env_kwargs):
    env = DummyVecEnv([make_env(rank=0, seed=seed, **env_kwargs)])

    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    return env


def load_adapter(spec):
    model_path = first_existing_path(spec.model_paths)
    vecnorm_path = first_existing_path(spec.vecnorm_paths)

    if model_path is None:
        print(f"[SKIP] {spec.name}: missing model.")
        return None, None, None

    if vecnorm_path is None:
        print(f"[SKIP] {spec.name}: missing VecNormalize file.")
        return None, None, None

    if spec.recurrent:
        if RecurrentPPO is None:
            print(f"[SKIP] {spec.name}: install sb3-contrib first.")
            return None, None, None

        model = RecurrentPPO.load(model_path)
        return RecurrentPPOAdapter(spec.name, model), model_path, vecnorm_path

    model = spec.model_class.load(model_path)
    return SB3AgentAdapter(spec.name, model), model_path, vecnorm_path


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


def plot_metric(summary_rows, metric, out_dir):
    names = [row["variant"] for row in summary_rows]
    means = [row[f"{metric}_mean"] for row in summary_rows]
    stds = [row[f"{metric}_std"] for row in summary_rows]

    plt.figure(figsize=(14, 5))
    plt.bar(names, means, yerr=stds, capsize=4)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric.replace("_", " "))
    plt.title(f"Variant Comparison: {metric.replace('_', ' ')}")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, f"{metric}.png")
    plt.savefig(path, dpi=200)
    plt.close()


def normalize_higher_is_better(values):
    values = np.array(values, dtype=float)

    if np.max(values) == np.min(values):
        return np.ones_like(values)

    return (values - np.min(values)) / (np.max(values) - np.min(values))


def normalize_lower_is_better(values):
    values = np.array(values, dtype=float)

    if np.max(values) == np.min(values):
        return np.ones_like(values)

    return (np.max(values) - values) / (np.max(values) - np.min(values))


def add_overall_scores(summary_rows):
    if not summary_rows:
        return summary_rows

    returns = [row["return_mean"] for row in summary_rows]
    costs = [row["cost_mean"] for row in summary_rows]
    drops = [row["dropped_requests_mean"] for row in summary_rows]
    latency = [row["latency_proxy_mean"] for row in summary_rows]
    stability = [row["action_stability_mean"] for row in summary_rows]

    return_score = normalize_higher_is_better(returns)
    cost_score = normalize_lower_is_better(costs)
    drop_score = normalize_lower_is_better(drops)
    latency_score = normalize_lower_is_better(latency)
    stability_score = normalize_higher_is_better(stability)

    for i, row in enumerate(summary_rows):
        row["overall_score"] = float(
            0.35 * return_score[i]
            + 0.25 * drop_score[i]
            + 0.15 * latency_score[i]
            + 0.15 * cost_score[i]
            + 0.10 * stability_score[i]
        )

    return summary_rows


def plot_overall_score(summary_rows, out_dir):
    names = [row["variant"] for row in summary_rows]
    scores = [row["overall_score"] for row in summary_rows]

    plt.figure(figsize=(14, 5))
    plt.bar(names, scores)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("overall score")
    plt.title("Variant Comparison: Overall Score")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, "overall_score.png")
    plt.savefig(path, dpi=200)
    plt.close()


def plot_best_family_models(best_by_family, out_dir):
    names = list(best_by_family.keys())
    returns = [best_by_family[name]["return_mean"] for name in names]
    costs = [best_by_family[name]["cost_mean"] for name in names]
    drops = [best_by_family[name]["dropped_requests_mean"] for name in names]

    x = np.arange(len(names))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, returns, width, label="return")
    plt.bar(x, costs, width, label="cost")
    plt.bar(x + width, drops, width, label="dropped requests")
    plt.xticks(x, names)
    plt.title("Best Final Model From Each Family")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "best_family_models.png")
    plt.savefig(path, dpi=200)
    plt.close()


def find_best_by_family(summary_rows):
    a2c_rows = [row for row in summary_rows if row["family"] == "A2C"]
    lstm_rows = [row for row in summary_rows if row["family"] == "PPO-LSTM"]

    best_by_family = {}

    if a2c_rows:
        best_by_family["A2C"] = max(a2c_rows, key=lambda row: row["overall_score"])

    if lstm_rows:
        best_by_family["PPO-LSTM"] = max(lstm_rows, key=lambda row: row["overall_score"])

    return best_by_family


def find_best_variants(summary_rows, best_by_family):
    if not summary_rows:
        return {}

    return {
        "best_a2c_by_overall_score": best_by_family.get("A2C"),
        "best_ppo_lstm_by_overall_score": best_by_family.get("PPO-LSTM"),
        "best_return_closer_to_zero": max(summary_rows, key=lambda row: row["return_mean"]),
        "best_lowest_cost": min(summary_rows, key=lambda row: row["cost_mean"]),
        "best_lowest_dropped_requests": min(summary_rows, key=lambda row: row["dropped_requests_mean"]),
        "best_lowest_latency_proxy": min(summary_rows, key=lambda row: row["latency_proxy_mean"]),
        "best_highest_action_stability": max(summary_rows, key=lambda row: row["action_stability_mean"]),
    }


def save_best_family_model(best_row, final_model_dir):
    os.makedirs(final_model_dir, exist_ok=True)

    model_src = best_row["model_path"]
    vecnorm_src = best_row["vecnorm_path"]

    model_dst = os.path.join(final_model_dir, "best_model.zip")
    vecnorm_dst = os.path.join(final_model_dir, "vecnormalize.pkl")
    metadata_dst = os.path.join(final_model_dir, "best_model_metadata.json")

    shutil.copy2(model_src, model_dst)
    shutil.copy2(vecnorm_src, vecnorm_dst)

    metadata = {
        "selected_variant": best_row["variant"],
        "family": best_row["family"],
        "selection_rule": "best_overall_score_within_same_family",
        "original_model_path": model_src,
        "original_vecnorm_path": vecnorm_src,
        "saved_model_path": model_dst,
        "saved_vecnorm_path": vecnorm_dst,
        "return_mean": best_row["return_mean"],
        "return_std": best_row["return_std"],
        "cost_mean": best_row["cost_mean"],
        "cost_std": best_row["cost_std"],
        "dropped_requests_mean": best_row["dropped_requests_mean"],
        "dropped_requests_std": best_row["dropped_requests_std"],
        "latency_proxy_mean": best_row["latency_proxy_mean"],
        "latency_proxy_std": best_row["latency_proxy_std"],
        "action_stability_mean": best_row["action_stability_mean"],
        "action_stability_std": best_row["action_stability_std"],
        "overall_score": best_row["overall_score"],
    }

    write_json(metadata_dst, metadata)

    print("\n=== Saved Best Family Model ===")
    print(f"Family:           {best_row['family']}")
    print(f"Selected variant: {best_row['variant']}")
    print(f"Model saved to:   {model_dst}")
    print(f"VecNormalize:     {vecnorm_dst}")
    print(f"Metadata:         {metadata_dst}")


def print_summary(summary_rows, best_by_family):
    print("\n=== Variant Comparison ===")

    for row in summary_rows:
        print(f"\n{row['variant']}")
        print(f"  Return:           {row['return_mean']:.2f} +/- {row['return_std']:.2f}")
        print(f"  Latency proxy:    {row['latency_proxy_mean']:.4f} +/- {row['latency_proxy_std']:.4f}")
        print(f"  Cost:             {row['cost_mean']:.2f} +/- {row['cost_std']:.2f}")
        print(f"  Dropped requests: {row['dropped_requests_mean']:.2f} +/- {row['dropped_requests_std']:.2f}")
        print(f"  Action stability: {row['action_stability_mean']:.4f} +/- {row['action_stability_std']:.4f}")
        print(f"  Overall score:    {row['overall_score']:.4f}")

    print("\n=== Best Final Models ===")
    for family, row in best_by_family.items():
        print(f"{family}: {row['variant']}")


def main():
    parser = argparse.ArgumentParser(description="Compare A2C and PPO-LSTM variants")

    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated evaluation seeds.",
    )

    parser.add_argument(
        "--traffic-profile",
        type=str,
        default="standard",
        choices=["standard", "deterministic", "poisson_only", "bursty_spikes"],
        help="Traffic profile used during evaluation.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="./results/variant_comparison",
        help="Folder where CSV, JSON, and plots will be saved.",
    )

    parser.add_argument(
        "--best-a2c-dir",
        type=str,
        default="./models/best_final_a2c",
        help="Folder where the selected best A2C model will be copied.",
    )

    parser.add_argument(
        "--best-recurrent-ppo-dir",
        type=str,
        default="./models/best_final_recurrent_ppo",
        help="Folder where the selected best PPO-LSTM model will be copied.",
    )

    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    env_kwargs = get_env_kwargs(args.traffic_profile)
    out_dir = os.path.join(args.out_dir, args.traffic_profile)

    all_episode_rows = []
    summary_rows = []

    for spec in VARIANT_SPECS:
        adapter, model_path, vecnorm_path = load_adapter(spec)

        if adapter is None:
            continue

        print(f"\nEvaluating {spec.name}...")
        print(f"  Model: {model_path}")
        print(f"  VecNormalize: {vecnorm_path}")

        rows_for_variant = []

        for seed in seeds:
            env = make_eval_env(
                seed=seed,
                vecnorm_path=vecnorm_path,
                env_kwargs=env_kwargs,
            )

            metrics = run_episode(adapter, env)
            env.close()

            row = {
                "variant": spec.name,
                "family": spec.family,
                "seed": seed,
                "traffic_profile": args.traffic_profile,
                "model_path": model_path,
                "vecnorm_path": vecnorm_path,
                **metrics,
            }

            all_episode_rows.append(row)
            rows_for_variant.append(row)

        summary = summarize(rows_for_variant)
        summary["variant"] = spec.name
        summary["family"] = spec.family
        summary["num_seeds"] = len(seeds)
        summary["traffic_profile"] = args.traffic_profile
        summary["model_path"] = model_path
        summary["vecnorm_path"] = vecnorm_path
        summary_rows.append(summary)

    summary_rows = add_overall_scores(summary_rows)
    best_by_family = find_best_by_family(summary_rows)
    best_variants = find_best_variants(summary_rows, best_by_family)

    os.makedirs(out_dir, exist_ok=True)

    write_csv(os.path.join(out_dir, "episode_results.csv"), all_episode_rows)
    write_csv(os.path.join(out_dir, "summary_results.csv"), summary_rows)
    write_json(os.path.join(out_dir, "summary_results.json"), summary_rows)
    write_json(os.path.join(out_dir, "best_variants.json"), best_variants)

    for metric in [
        "return",
        "latency_proxy",
        "cost",
        "dropped_requests",
        "action_stability",
    ]:
        plot_metric(summary_rows, metric, out_dir)

    plot_overall_score(summary_rows, out_dir)
    plot_best_family_models(best_by_family, out_dir)

    if "A2C" in best_by_family:
        save_best_family_model(
            best_row=best_by_family["A2C"],
            final_model_dir=args.best_a2c_dir,
        )

    if "PPO-LSTM" in best_by_family:
        save_best_family_model(
            best_row=best_by_family["PPO-LSTM"],
            final_model_dir=args.best_recurrent_ppo_dir,
        )

    print_summary(summary_rows, best_by_family)
    print(f"\nSaved results to: {out_dir}")
    print(f"Saved best A2C model to: {args.best_a2c_dir}")
    print(f"Saved best PPO-LSTM model to: {args.best_recurrent_ppo_dir}")


if __name__ == "__main__":
    main()