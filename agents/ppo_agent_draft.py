"""
PPO agent wrapper around Stable-Baselines3 for real-time ad bidding.

This module does NOT implement PPO from scratch. Instead, it provides a thin,
project-friendly adapter around stable_baselines3.PPO so the rest of the codebase
can interact with a consistent agent interface.

Expected usage:
    env = AuctionNetGymEnv(env_config)
    agent = PPOAgent(env.observation_space, env.action_space, agent_config, env=env)
    agent.update(total_timesteps=50_000)
    action, _, _ = agent.select_action(obs)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from stable_baselines3 import PPO


class PPOAgent:
    """Thin wrapper around stable_baselines3.PPO."""

    def __init__(
        self,
        observation_space,
        action_space,
        config: dict,
        env=None,
    ):
        """
        Initialize the SB3 PPO model.

        Args:
            observation_space: Gymnasium observation space. Kept for interface
                consistency, even though SB3 primarily needs `env`.
            action_space: Gymnasium action space. Kept for interface consistency.
            config (dict): Hyperparameters from configs/default.yaml under 'agent'.
            env: Gymnasium-compatible environment or VecEnv. Required for training.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        self.env = env

        self.policy = config.get("policy", "MlpPolicy")
        self.verbose = int(config.get("verbose", 1))
        self.device = config.get("device", "auto")
        self.tensorboard_log = config.get("tensorboard_log", None)

        # Core PPO hyperparameters
        self.learning_rate = float(config.get("lr", 3e-4))
        self.n_steps = int(config.get("n_steps", 2048))
        self.batch_size = int(config.get("batch_size", 64))
        self.n_epochs = int(config.get("n_epochs", 10))
        self.gamma = float(config.get("gamma", 0.99))
        self.gae_lambda = float(config.get("gae_lambda", 0.95))
        self.clip_range = float(config.get("clip_range", 0.2))
        self.ent_coef = float(config.get("ent_coef", 0.0))
        self.vf_coef = float(config.get("vf_coef", 0.5))
        self.max_grad_norm = float(config.get("max_grad_norm", 0.5))
        self.seed = config.get("seed", None)

        # Optional policy network structure
        policy_kwargs = config.get("policy_kwargs", None)

        self.model: Optional[PPO] = None
        if env is not None:
            self.model = PPO(
                policy=self.policy,
                env=env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm,
                verbose=self.verbose,
                seed=self.seed,
                device=self.device,
                tensorboard_log=self.tensorboard_log,
                policy_kwargs=policy_kwargs,
            )

    def set_env(self, env) -> None:
        """Attach or replace the environment."""
        self.env = env
        if self.model is None:
            self.model = PPO(
                policy=self.policy,
                env=env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm,
                verbose=self.verbose,
                seed=self.seed,
                device=self.device,
                tensorboard_log=self.tensorboard_log,
                policy_kwargs=self.config.get("policy_kwargs", None),
            )
        else:
            self.model.set_env(env)

    def select_action(self, observation, deterministic: bool = False):
        """
        Predict an action from the current policy.

        Args:
            observation (np.ndarray): Current observation.
            deterministic (bool): Whether to use deterministic policy output.

        Returns:
            action (np.ndarray): Predicted action.
            log_prob: Not exposed directly by SB3's public `predict` API, so None.
            value: Not exposed directly by SB3's public `predict` API, so None.
        """
        if self.model is None:
            raise ValueError("PPO model is not initialized. Call set_env(env) first.")

        obs = np.asarray(observation, dtype=np.float32)
        action, _states = self.model.predict(obs, deterministic=deterministic)
        return action, None, None

    def update(self, rollout_buffer=None, total_timesteps: Optional[int] = None, **learn_kwargs):
        """
        Train the SB3 PPO model.

        Since PPO is delegated to SB3, `rollout_buffer` is ignored. Training is
        performed by calling `learn()`.

        Args:
            rollout_buffer: Ignored. Kept only for backward compatibility.
            total_timesteps (int): Number of timesteps to train for.
            **learn_kwargs: Extra keyword arguments passed to `model.learn()`.

        Returns:
            dict: Minimal diagnostics.
        """
        if self.model is None:
            raise ValueError("PPO model is not initialized. Call set_env(env) first.")

        if total_timesteps is None:
            total_timesteps = int(self.config.get("total_timesteps", 10_000))

        self.model.learn(total_timesteps=total_timesteps, **learn_kwargs)
        return {"trained_timesteps": total_timesteps}

    def save(self, path: str):
        """
        Save the SB3 model checkpoint.

        Args:
            path (str): Save path, e.g. 'saved_models/ppo_agent'.
                SB3 will add '.zip' if needed.
        """
        if self.model is None:
            raise ValueError("PPO model is not initialized. Nothing to save.")

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))

    def load(self, path: str, env=None):
        """
        Load a saved SB3 PPO model.

        Args:
            path (str): Path to checkpoint.
            env: Optional env to attach after loading.

        Returns:
            PPOAgent: self
        """
        if env is not None:
            self.env = env

        self.model = PPO.load(path, env=self.env, device=self.device)
        return self