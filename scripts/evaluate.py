import argparse
from html import parser
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure project root is importable when running:
# python scripts/evaluate.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.gym_wrapper import AuctionNetGymEnv
from agents.ppo_agent import PPOAgent
from evaluation.metrics import compute_all_metrics
from agents.iql_agent import IQLAgent
from agents.fixed_bid_baseline import FixedBidBaseline

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO on AuctionNetGymEnv")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_local.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default = None,
        help="Path to saved PPO checkpoint (.zip or base path)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy during evaluation",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default="ppo",
        choices=["ppo", "fixed", "iql"],
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="eval",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_config(config_path: str) -> dict:
    path = resolve_path(config_path)
    print(f"Loading config from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(env_config: dict):
    def _init():
        env = AuctionNetGymEnv(env_config)
        env = Monitor(env)
        return env
    return _init


def rollout_one_episode(vec_env, agent, deterministic: bool = False):
    """
    Roll out one full episode and collect a trajectory.

    Stored step format:
        (obs, action, reward, info)
    which matches evaluation.metrics expectations.
    """
    obs = vec_env.reset()
    trajectory = []

    done = False
    while not done:
        action, _, _ = agent.select_action(obs, deterministic=deterministic)
        next_obs, rewards, dones, infos = vec_env.step(action)

        reward_scalar = float(rewards[0])
        done = bool(dones[0])

        # Keep info as returned by VecEnv (list[dict]); metrics.py can unwrap it.
        trajectory.append((obs, action, reward_scalar, infos))

        obs = next_obs

    return trajectory


def summarize_metrics(metrics_list: list[dict]) -> dict:
    keys = metrics_list[0].keys()
    summary = {}
    for key in keys:
        values = np.array([m[key] for m in metrics_list], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def print_episode_metrics(ep_idx: int, metrics: dict):
    print(f"\nEpisode {ep_idx + 1}")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.6f}")


def print_summary(summary: dict, n_episodes: int):
    print("\n" + "=" * 60)
    print(f"Evaluation summary over {n_episodes} episode(s)")
    print("=" * 60)
    for key, stats in summary.items():
        print(
            f"{key:20s} "
            f"mean={stats['mean']:.6f}  "
            f"std={stats['std']:.6f}  "
            f"min={stats['min']:.6f}  "
            f"max={stats['max']:.6f}"
        )


def main():
    args = parse_args()
    config = load_config(args.config)

    env_config = config["environment"]
    agent_config = config.get("agent", {})

    checkpoint_path = None

    if args.agent_type == "ppo":
        if args.checkpoint is None:
            raise ValueError("PPO evaluation requires --checkpoint.")

        checkpoint_path = resolve_path(args.checkpoint)

        if not checkpoint_path.exists():
            zip_path = checkpoint_path.with_suffix(".zip")
            if zip_path.exists():
                checkpoint_path = zip_path
            else:
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Using checkpoint: {checkpoint_path}")

    vec_env = DummyVecEnv([make_env(env_config)])

    if args.agent_type == "ppo":
        if args.checkpoint is None:
            raise ValueError("PPO requires --checkpoint")

        agent = PPOAgent(
            observation_space=vec_env.observation_space,
            action_space=vec_env.action_space,
            config=agent_config,
            env=vec_env,
        )
        agent.load(str(checkpoint_path), env=vec_env)

    elif args.agent_type == "fixed":
        agent = FixedBidBaseline(
            observation_space=vec_env.observation_space,
            action_space=vec_env.action_space,
            config=agent_config,
            env=vec_env,
        )
    elif args.agent_type == "iql":
        agent = IQLAgent(
            observation_space=vec_env.observation_space,
            action_space=vec_env.action_space,
            config=agent_config,
            env=vec_env,
        )
    all_metrics = []

    for ep in range(args.episodes):
        trajectory = rollout_one_episode(
            vec_env=vec_env,
            agent=agent,
            deterministic=args.deterministic,
        )
        metrics = compute_all_metrics(trajectory)
        all_metrics.append(metrics)
        print_episode_metrics(ep, metrics)

    summary = summarize_metrics(all_metrics)
    print_summary(summary, args.episodes)


if __name__ == "__main__":
    main()