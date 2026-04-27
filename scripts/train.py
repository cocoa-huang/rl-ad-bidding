import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure project root is importable when running:
# python scripts/train.py
# or:
# python -m scripts.train
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.gym_wrapper import AuctionNetGymEnv
from agents.ppo_agent import PPOAgent
from agents.fixed_bid_baseline import FixedBidBaseline
from agents.iql_agent import IQLAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate-compatible agents on AuctionNetGymEnv")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_local.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default=None,
        choices=["ppo", "fixed", "iql"],
        help="Agent type. If omitted, uses agent.algorithm from config.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name override",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help=(
            "Optional training budget. For PPO this is env timesteps; "
            "for IQL this is gradient updates."
        ),
    )
    parser.add_argument(
        "--offline-dataset",
        type=str,
        default=None,
        help="Path to offline transition dataset for IQL, usually .npz.",
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


def infer_total_timesteps(config: dict, cli_total_timesteps: Optional[int]) -> int:
    if cli_total_timesteps is not None:
        return int(cli_total_timesteps)

    training_cfg = config.get("training", {})
    env_cfg = config.get("environment", {})

    if "total_timesteps" in training_cfg:
        return int(training_cfg["total_timesteps"])

    n_episodes = int(training_cfg.get("n_episodes", 1000))
    num_ticks = int(env_cfg.get("num_ticks", 48))
    return n_episodes * num_ticks


def get_agent_type(args, config: dict) -> str:
    if args.agent_type is not None:
        return args.agent_type
    return str(config.get("agent", {}).get("algorithm", "ppo")).lower()


def load_offline_dataset(path_str: str) -> dict:
    """
    Load an offline transition dataset for IQL.

    Expected .npz keys:
        observations or obs
        actions
        rewards
        next_observations or next_obs
        dones
    """
    path = resolve_path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Offline dataset not found: {path}")

    if path.suffix.lower() != ".npz":
        raise ValueError(
            f"Unsupported offline dataset format: {path.suffix}. "
            "Use a .npz file with observations/actions/rewards/next_observations/dones."
        )

    data = np.load(path, allow_pickle=False)
    dataset = {key: data[key] for key in data.files}

    aliases = {
        "observations": ["observations", "obs"],
        "actions": ["actions"],
        "rewards": ["rewards"],
        "next_observations": ["next_observations", "next_obs"],
        "dones": ["dones"],
    }
    missing = []
    for canonical, candidates in aliases.items():
        if not any(name in dataset for name in candidates):
            missing.append(canonical)
    if missing:
        raise KeyError(f"Offline dataset missing required fields: {missing}. Found keys: {list(dataset)}")

    print(f"Loaded offline dataset from: {path}")
    for key, value in dataset.items():
        print(f"  {key:20s}: shape={value.shape}, dtype={value.dtype}")
    return dataset


def build_agent(agent_type: str, observation_space, action_space, agent_config: dict, env=None):
    if agent_type == "ppo":
        return PPOAgent(observation_space, action_space, agent_config, env=env)
    if agent_type == "fixed":
        return FixedBidBaseline(observation_space, action_space, agent_config, env=env)
    if agent_type == "iql":
        return IQLAgent(observation_space, action_space, agent_config, env=env)
    raise ValueError(f"Unknown agent_type: {agent_type}")


def train_ppo(agent, config: dict, args, run_name: str):
    env_config = config["environment"]
    training_config = config.get("training", {})
    total_timesteps = infer_total_timesteps(config, args.total_timesteps)

    save_interval = int(training_config.get("save_interval", 500))
    num_ticks = int(env_config.get("num_ticks", 48))
    save_freq_steps = max(save_interval * num_ticks, 1)

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_steps,
        save_path=str(PROJECT_ROOT / "saved_models"),
        name_prefix=f"ppo_{run_name}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    print("=" * 60)
    print("Agent type:       ppo")
    print(f"Config:           {args.config}")
    print(f"Run name:         {run_name}")
    print(f"Total timesteps:  {total_timesteps}")
    print(f"Save every:       {save_freq_steps} env steps")
    print("=" * 60)

    metrics = agent.update(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    final_path = PROJECT_ROOT / "saved_models" / f"ppo_{run_name}_final"
    agent.save(str(final_path))
    print(f"Training complete. Metrics: {metrics}")
    print(f"Final PPO model saved to: {final_path}.zip")


def train_fixed(agent, args, run_name: str):
    """No-op training for fixed baseline; saves its config for reproducibility."""
    print("=" * 60)
    print("Agent type:       fixed")
    print(f"Config:           {args.config}")
    print(f"Run name:         {run_name}")
    print("Training:         skipped (fixed baseline has no learnable parameters)")
    print("=" * 60)

    final_path = PROJECT_ROOT / "saved_models" / f"fixed_{run_name}.json"
    agent.save(str(final_path))
    print(f"Fixed baseline config saved to: {final_path}")


def train_iql(agent, config: dict, args, run_name: str):
    training_config = config.get("training", {})
    total_updates = infer_total_timesteps(config, args.total_timesteps)

    dataset_path = (
        args.offline_dataset
        or training_config.get("offline_dataset")
        or training_config.get("offline_dataset_path")
    )
    if dataset_path is None:
        raise ValueError(
            "IQL requires an offline dataset. Provide --offline-dataset path/to/data.npz "
            "or set training.offline_dataset_path in the config."
        )

    offline_dataset = load_offline_dataset(dataset_path)

    print("=" * 60)
    print("Agent type:       iql")
    print(f"Config:           {args.config}")
    print(f"Run name:         {run_name}")
    print(f"Gradient updates: {total_updates}")
    print(f"Offline dataset:  {resolve_path(dataset_path)}")
    print("=" * 60)

    metrics = agent.update(
        offline_dataset=offline_dataset,
        total_timesteps=total_updates,
    )

    final_path = PROJECT_ROOT / "saved_models" / f"iql_{run_name}_final"
    agent.save(str(final_path))
    print(f"Training complete. Metrics: {metrics}")
    print(f"Final IQL model saved to: {final_path.with_suffix('.pt')}")


def main():
    args = parse_args()
    config = load_config(args.config)

    env_config = config["environment"]
    agent_config = config.get("agent", {})
    logging_config = config.get("logging", {})

    agent_type = get_agent_type(args, config)
    run_name = args.run_name or logging_config.get("run_name", f"{agent_type}_run")

    os.makedirs(PROJECT_ROOT / "saved_models", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "logs", exist_ok=True)

    vec_env = DummyVecEnv([make_env(env_config)])
    agent = build_agent(
        agent_type=agent_type,
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        agent_config=agent_config,
        env=vec_env,
    )

    if agent_type == "ppo":
        train_ppo(agent, config, args, run_name)
    elif agent_type == "fixed":
        train_fixed(agent, args, run_name)
    elif agent_type == "iql":
        train_iql(agent, config, args, run_name)
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")


if __name__ == "__main__":
    main()
