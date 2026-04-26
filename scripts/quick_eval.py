"""
Quick evaluation of a saved PPO checkpoint, or a fixed-alpha baseline.

Usage:
    python scripts/quick_eval.py --run-name gcp-run-3 [--n-episodes 10]
    python scripts/quick_eval.py --config configs/gcp-run-5.yaml --alpha-override 100
"""
import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3.common.monitor import Monitor  # needed for vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.ppo_agent import PPOAgent
from environment.gym_wrapper import AuctionNetGymEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None,
                        help="Saved checkpoint name (required unless --alpha-override is set)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--alpha-override", type=float, default=None,
                        help="If set, bypass the policy and emit this constant alpha "
                             "(in [0, max_bid_multiplier]) every step. Used for "
                             "fixed-bid baseline diagnostics.")
    return parser.parse_args()


def _alpha_to_action(alpha: float, max_bid_multiplier: float) -> np.ndarray:
    """Invert the env's action→alpha rescaling: alpha = ((a+1)/2) * max_bid_multiplier."""
    a = (alpha / max_bid_multiplier) * 2.0 - 1.0
    return np.array([a], dtype=np.float32)


def evaluate_fixed_alpha(env: AuctionNetGymEnv, alpha: float, n_episodes: int) -> dict:
    """Roll out n_episodes with a constant alpha. Returns same keys as PPOAgent.evaluate
    plus ep_rew_mean (computed from the env's own reward signal).
    """
    if not 0 <= alpha <= env.max_bid_multiplier:
        raise ValueError(f"alpha {alpha} outside [0, {env.max_bid_multiplier}]")
    action = _alpha_to_action(alpha, env.max_bid_multiplier)

    total_conversion_value = 0.0
    total_cost = 0.0
    total_won = 0
    total_auctions = 0
    ep_rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        ep_rew = 0.0
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            ep_rew += reward
            total_conversion_value += info["conversion_value"]
            total_cost += info["cost"]
            total_won += info["won"]
            total_auctions += info.get("total_pvs", env._pv_gen.pv_values[0].shape[0])
        ep_rewards.append(ep_rew)

    return {
        "roi": total_conversion_value / total_cost if total_cost > 0 else 0.0,
        "budget_utilization": total_cost / (n_episodes * env.budget),
        "win_rate": total_won / total_auctions if total_auctions > 0 else 0.0,
        "ep_rew_mean": statistics.mean(ep_rewards),
        "ep_rew_std": statistics.stdev(ep_rewards) if len(ep_rewards) > 1 else 0.0,
    }


def main():
    args = parse_args()

    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    env_config = config["environment"]
    agent_config = config.get("agent", {})

    if args.alpha_override is not None:
        raw_env = AuctionNetGymEnv(env_config)
        print(f"Fixed-alpha baseline: α={args.alpha_override}, "
              f"{args.n_episodes} episodes, reward = "
              f"{env_config.get('reward_value_scale', 1.0)}*v "
              f"- {env_config.get('reward_lambda_cost', 1.0)}*c "
              f"+ {env_config.get('reward_beta_pacing', 0.0)}*min(pacing,1)")
        metrics = evaluate_fixed_alpha(raw_env, args.alpha_override, args.n_episodes)
        print(f"  ROI:                {metrics['roi']:.4f}")
        print(f"  Budget utilization: {metrics['budget_utilization']:.2%}")
        print(f"  Win rate:           {metrics['win_rate']:.2%}")
        print(f"  ep_rew_mean:        {metrics['ep_rew_mean']:.2f} ± {metrics['ep_rew_std']:.2f}")
        return

    if args.run_name is None:
        print("Either --run-name or --alpha-override is required")
        sys.exit(1)

    model_path = PROJECT_ROOT / "saved_models" / f"ppo_{args.run_name}_best" / "best_model"
    if not model_path.with_suffix(".zip").exists():
        print(f"No checkpoint found at {model_path}.zip")
        sys.exit(1)

    vec_env = DummyVecEnv([lambda: Monitor(AuctionNetGymEnv(env_config))])
    vecnorm_path = model_path.with_name("best_model_vecnormalize.pkl")
    if vecnorm_path.exists():
        vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"Loaded VecNormalize stats: {vecnorm_path}")
    elif config.get("training", {}).get("norm_reward", False):
        print(f"WARNING: expected VecNormalize stats but did not find {vecnorm_path}")

    raw_env = AuctionNetGymEnv(env_config)

    agent = PPOAgent(
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        config=agent_config,
        env=vec_env,
    )
    agent.load(str(model_path), env=vec_env)

    print(f"Evaluating {args.run_name} over {args.n_episodes} episodes...")
    metrics = agent.evaluate(raw_env, n_eval_episodes=args.n_episodes)
    print(f"  ROI:                {metrics['roi']:.4f}")
    print(f"  Budget utilization: {metrics['budget_utilization']:.2%}")
    print(f"  Win rate:           {metrics['win_rate']:.2%}")
    print(f"  ep_rew_mean:        {metrics['ep_rew_mean']:.2f} ± {metrics['ep_rew_std']:.2f}")


if __name__ == "__main__":
    main()
