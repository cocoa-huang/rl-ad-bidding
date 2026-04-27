import os
import sys
from pathlib import Path
import numpy as np
import yaml
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.gym_wrapper import AuctionNetGymEnv
from agents.fixed_bid_baseline import FixedBidBaseline


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env(env_config):
    def _init():
        env = AuctionNetGymEnv(env_config)
        env = Monitor(env)
        return env
    return _init


def collect_dataset(config_path, save_path, num_episodes=10):
    config = load_config(config_path)

    env_config = config["environment"]
    agent_config = config["agent"]

    vec_env = DummyVecEnv([make_env(env_config)])

    agent = FixedBidBaseline(
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        config=agent_config,
        env=vec_env,
    )

    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    for ep in range(num_episodes):
        obs = vec_env.reset()
        done = False

        while not done:
            action, _, _ = agent.select_action(obs)
            next_obs, reward, done_arr, infos = vec_env.step(action)

            observations.append(obs.copy())
            actions.append(action.copy())
            rewards.append(reward.copy())
            next_observations.append(next_obs.copy())
            dones.append(done_arr.copy())

            obs = next_obs
            done = bool(done_arr[0])

    dataset = {
        "observations": np.array(observations),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "next_observations": np.array(next_observations),
        "dones": np.array(dones),
    }

    os.makedirs(Path(save_path).parent, exist_ok=True)
    np.savez_compressed(save_path, **dataset)

    print(f"Saved dataset to {save_path}")
    print("Shapes:")
    for k, v in dataset.items():
        print(f"{k}: {v.shape}")


if __name__ == "__main__":
    collect_dataset(
        config_path="configs/fixed.yaml",
        save_path="data/offline_dataset_debug.npz",
        num_episodes=5,
    )