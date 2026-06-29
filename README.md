# copy-repo

import numpy as np
import argparse
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_factory import make_env
from baseline_agent import RuleBasedBaseline
import os

def evaluate_agent(model, vecnorm_path=None, n_episodes=10, seed=42):
    """Deterministic policy evaluation. Returns mean/std of total_reward,
    operational_cost, dropped_requests, queue_occupancy_rate."""
    env = DummyVecEnv([make_env(rank=100, seed=seed)])
    if vecnorm_path and os.path.exists(vecnorm_path):  # load TRAINING normalization stats
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False               # do not update stats
        env.norm_reward = False            # report real rewards
        
    rewards, costs, drops, qocc = [], [], [], []
    for ep in range(n_episodes):
        obs = env.reset()
        done = [False]
        R = c = d = q = 0
        steps = 0
        while not done[0]:
            if model == "random":
                action = np.array([env.action_space.sample()])
            elif hasattr(model, 'predict'):
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.array([1]) # default hold if unknown
                
            obs, r, done, info = env.step(action)
            R += r[0]
            c += info[0]["active"]
            d += info[0]["dropped"]
            q += info[0]["queue"]
            steps += 1
        rewards.append(R)
        costs.append(c)
        drops.append(d)
        qocc.append(q / (steps * 500))     # mean fractional queue occupancy
        
    agg = lambda x: (float(np.mean(x)), float(np.std(x)))
    
    # ensure env is properly closed
    env.close()
    
    return {"reward": agg(rewards), "cost": agg(costs),
            "dropped": agg(drops), "queue_occ": agg(qocc)}

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained policies deterministically")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for env")
    args = parser.parse_args()

    results = {}

    print("Evaluating Baseline...")
    baseline_model = RuleBasedBaseline()
    results["Baseline"] = evaluate_agent(baseline_model, vecnorm_path=None, n_episodes=args.episodes, seed=args.seed)

    print("Evaluating Random...")
    results["Random"] = evaluate_agent("random", vecnorm_path=None, n_episodes=args.episodes, seed=args.seed)

    print("Evaluating PPO...")
    if os.path.exists("./models/best_ppo/best_model.zip") and os.path.exists("./models/vecnormalize_ppo.pkl"):
        ppo_model = PPO.load("./models/best_ppo/best_model.zip")
        results["PPO"] = evaluate_agent(ppo_model, "./models/vecnormalize_ppo.pkl", args.episodes, args.seed)
    else:
        print("PPO model or vecnormalize not found. Skipping.")

    print("Evaluating DQN...")
    if os.path.exists("./models/best_dqn/best_model.zip") and os.path.exists("./models/vecnormalize_dqn.pkl"):
        dqn_model = DQN.load("./models/best_dqn/best_model.zip")
        results["DQN"] = evaluate_agent(dqn_model, "./models/vecnormalize_dqn.pkl", args.episodes, args.seed)
    else:
        print("DQN model or vecnormalize not found. Skipping.")

    print("\n--- Final Evaluation Results ---")
    for agent, metrics in results.items():
        print(f"\n{agent}:")
        print(f"  Reward:      {metrics['reward'][0]:.2f} ± {metrics['reward'][1]:.2f}")
        print(f"  Cost:        {metrics['cost'][0]:.2f} ± {metrics['cost'][1]:.2f}")
        print(f"  Dropped:     {metrics['dropped'][0]:.2f} ± {metrics['dropped'][1]:.2f}")
        print(f"  Queue Occ:   {metrics['queue_occ'][0]:.4f} ± {metrics['queue_occ'][1]:.4f}")

if __name__ == "__main__":
    main()
