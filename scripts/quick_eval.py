"""
Quick evaluation of a saved PPO checkpoint.
Usage: python scripts/quick_eval.py --run-name gcp-run-3 [--n-episodes 10]
"""
import argparse
import sys
from pathlib import Path

import yaml
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.ppo_agent import PPOAgent
from environment.gym_wrapper import AuctionNetGymEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n-episodes", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    env_config = config["environment"]
    agent_config = config.get("agent", {})

    model_path = PROJECT_ROOT / "saved_models" / f"ppo_{args.run_name}_best" / "best_model"
    if not model_path.with_suffix(".zip").exists():
        print(f"No checkpoint found at {model_path}.zip")
        sys.exit(1)

    eval_env = DummyVecEnv([lambda: Monitor(AuctionNetGymEnv(env_config))])
    agent = PPOAgent(
        observation_space=eval_env.observation_space,
        action_space=eval_env.action_space,
        config=agent_config,
        env=eval_env,
    )
    agent.load(str(model_path), env=eval_env)

    print(f"Evaluating {args.run_name} over {args.n_episodes} episodes...")
    metrics = agent.evaluate(eval_env, n_episodes=args.n_episodes)
    print(f"  ROI:                {metrics['roi']:.4f}")
    print(f"  Budget utilization: {metrics['budget_utilization']:.2%}")
    print(f"  Win rate:           {metrics['win_rate']:.2%}")


if __name__ == "__main__":
    main()
