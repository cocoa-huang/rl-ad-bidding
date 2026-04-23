import argparse
import os
import sys
from pathlib import Path

import yaml
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Ensure project root is importable when running:
# python scripts/train.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.gym_wrapper import AuctionNetGymEnv
from agents.ppo_agent import PPOAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on AuctionNetGymEnv")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_local.yaml",
        help="Path to YAML config file",
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
        help="Optional override for total training timesteps",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()

    print(f"Loading config from: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(env_config: dict):
    def _init():
        env = AuctionNetGymEnv(env_config)
        env = Monitor(env)
        return env
    return _init


def infer_total_timesteps(config: dict, cli_total_timesteps: int | None) -> int:
    if cli_total_timesteps is not None:
        return cli_total_timesteps

    training_cfg = config.get("training", {})
    env_cfg = config.get("environment", {})

    if "total_timesteps" in training_cfg:
        return int(training_cfg["total_timesteps"])

    n_episodes = int(training_cfg.get("n_episodes", 1000))
    num_ticks = int(env_cfg.get("num_ticks", 48))
    return n_episodes * num_ticks


def main():
    args = parse_args()
    config = load_config(args.config)

    env_config = config["environment"]
    agent_config = config.get("agent", {})
    training_config = config.get("training", {})
    logging_config = config.get("logging", {})

    run_name = args.run_name or logging_config.get("run_name", "baseline")
    total_timesteps = infer_total_timesteps(config, args.total_timesteps)

    os.makedirs(PROJECT_ROOT / "saved_models", exist_ok=True)
    os.makedirs(PROJECT_ROOT / "logs", exist_ok=True)

    # Use multiprocessing to utilize Modal's 32 CPU cores
    n_envs = 16
    vec_env = SubprocVecEnv([make_env(env_config) for _ in range(n_envs)])

    agent = PPOAgent(
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        config=agent_config,
        env=vec_env,
    )

    # Automatically load the best previous model if it exists to warm-start training!
    best_model_path = PROJECT_ROOT / "saved_models" / f"ppo_{run_name}_best.zip"
    if best_model_path.exists():
        print(f"🔄 Found existing checkpoint at {best_model_path}! Loading to continue training...")
        agent.load(str(best_model_path), env=vec_env)
    else:
        print("✨ No existing checkpoint found. Starting fresh training run!")

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

    # Setup Early Stopping & Evaluation
    eval_freq_steps = max(int(training_config.get("eval_interval", 100)) * num_ticks, 1)
    eval_env = DummyVecEnv([make_env(env_config)])
    
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5, 
        min_evals=5, 
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(PROJECT_ROOT / "saved_models" / f"ppo_{run_name}_best"),
        log_path=str(PROJECT_ROOT / "logs"),
        eval_freq=eval_freq_steps,
        callback_after_eval=stop_train_callback,
        deterministic=True,
        render=False,
    )

    # Initialize WandB if enabled
    use_wandb = config.get("logging", {}).get("use_wandb", False)
    callbacks = [checkpoint_callback, eval_callback]
    
    if use_wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        
        wandb.init(
            project="rl-ad-bidding",
            name=run_name,
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        wandb_callback = WandbCallback(
            gradient_save_freq=100,
            model_save_path=str(PROJECT_ROOT / "saved_models" / f"ppo_{run_name}_wandb"),
            verbose=2,
        )
        callbacks.append(wandb_callback)

    print("=" * 60)
    print(f"Config:           {args.config}")
    print(f"Run name:         {run_name}")
    print(f"Total timesteps:  {total_timesteps}")
    print(f"Save every:       {save_freq_steps} env steps")
    print(f"WandB Logging:    {'Enabled' if use_wandb else 'Disabled'}")
    print("=" * 60)

    metrics = agent.update(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    if use_wandb:
        wandb.finish()

    final_path = PROJECT_ROOT / "saved_models" / f"ppo_{run_name}_final"
    agent.save(str(final_path))

    print(f"Training complete. Metrics: {metrics}")
    print(f"Final model saved to: {final_path}.zip")


if __name__ == "__main__":
    main()