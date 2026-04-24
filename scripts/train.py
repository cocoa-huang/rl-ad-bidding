import argparse
import os
import sys
from pathlib import Path

import yaml
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.gym_wrapper import AuctionNetGymEnv
from agents.ppo_agent import PPOAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on AuctionNetGymEnv")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path.resolve(), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(env_config: dict):
    def _init():
        return Monitor(AuctionNetGymEnv(env_config))
    return _init


def main():
    args = parse_args()
    config = load_config(args.config)

    env_config = config["environment"]
    agent_config = config.get("agent", {})
    training_config = config.get("training", {})
    logging_config = config.get("logging", {})

    run_name = args.run_name or logging_config.get("run_name", "baseline")

    total_timesteps = args.total_timesteps
    if total_timesteps is None:
        total_timesteps = int(agent_config.get("total_timesteps", 500_000))

    os.makedirs(PROJECT_ROOT / "saved_models", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "logs", exist_ok=True)

    # --- W&B (optional — controlled by logging.use_wandb in config) ---
    use_wandb = logging_config.get("use_wandb", False)
    wandb_callback = None
    if use_wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        wandb.init(
            project=logging_config.get("project_name", "rl-ad-bidding"),
            entity=logging_config.get("entity", None),
            name=run_name,
            config={**env_config, **agent_config, "total_timesteps": total_timesteps},
            sync_tensorboard=True,
        )
        wandb_callback = WandbCallback(
            gradient_save_freq=0,
            verbose=0,
        )

    # --- Environments ---
    n_envs = int(training_config.get("n_envs", 8))
    vec_env = SubprocVecEnv([make_env(env_config) for _ in range(n_envs)])
    eval_env = DummyVecEnv([make_env(env_config)])

    # --- Agent ---
    agent_config_train = {
        **agent_config,
        "tensorboard_log": str(PROJECT_ROOT / "logs"),
    }

    agent = PPOAgent(
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        config=agent_config_train,
        env=vec_env,
    )

    # --- Warm-start: resume from best checkpoint if available ---
    best_model_path = PROJECT_ROOT / "saved_models" / f"ppo_{run_name}_best" / "best_model.zip"
    if best_model_path.exists():
        print(f"Resuming from checkpoint: {best_model_path}")
        agent.load(str(best_model_path), env=vec_env)

    # --- Callbacks ---
    num_ticks = int(env_config.get("num_ticks", 48))
    save_interval = int(training_config.get("save_interval", 500))
    eval_interval = int(training_config.get("eval_interval", 100))
    early_stop_patience = int(training_config.get("early_stop_patience", 10))

    checkpoint_cb = CheckpointCallback(
        save_freq=max(save_interval * num_ticks, 1),
        save_path=str(PROJECT_ROOT / "saved_models"),
        name_prefix=f"ppo_{run_name}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    early_stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=early_stop_patience,
        min_evals=early_stop_patience,
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(PROJECT_ROOT / "saved_models" / f"ppo_{run_name}_best"),
        eval_freq=max(eval_interval * num_ticks, 1),
        n_eval_episodes=5,
        callback_after_eval=early_stop_cb,
        deterministic=True,
        verbose=1,
    )

    print("=" * 60)
    print(f"Run name:        {run_name}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"W&B:             {use_wandb}")
    print("=" * 60)

    callbacks = [checkpoint_cb, eval_cb]
    if wandb_callback is not None:
        callbacks.append(wandb_callback)

    agent.update(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    final_path = PROJECT_ROOT / "saved_models" / f"ppo_{run_name}_final"
    agent.save(str(final_path))
    print(f"Model saved to: {final_path}.zip")

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
