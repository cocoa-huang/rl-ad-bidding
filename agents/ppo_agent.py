"""
PPO agent wrapper around Stable-Baselines3 for real-time ad bidding.

This module does NOT implement PPO from scratch. Instead, it provides a thin,
project-friendly adapter around stable_baselines3.PPO so the rest of the codebase
can interact with a consistent agent interface.

Expected usage:
    env = AuctionNetGymEnv(env_config)
    agent = PPOAgent(env.observation_space, env.action_space, agent_config, env=env)
    agent.update(total_timesteps=500_000)
    action, _, _ = agent.select_action(obs)

Reference:
    Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
    Proximal Policy Optimization Algorithms. arXiv:1707.06347.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize


class PPOAgent:
    """Thin wrapper around stable_baselines3.PPO.

    Attributes:
        config (dict): Hyperparameters from configs/YAML under the 'agent' section.
        model (PPO | None): The underlying SB3 PPO model. None until an env is attached.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        config: dict,
        env=None,
    ):
        """Initialize the SB3 PPO model.

        Args:
            observation_space: Gymnasium observation space. Kept for interface
                consistency; SB3 derives input dims from `env` directly.
            action_space: Gymnasium action space. Kept for interface consistency.
            config (dict): Hyperparameters from configs/default.yaml under 'agent'.
            env: Gymnasium-compatible environment or VecEnv. Required for training.
                NOTE: The action space [0, max_bid_multiplier] is non-symmetric
                around 0 because competitors bid cpa * pValue with CPA in [60, 130],
                requiring our agent to reach up to 150. SB3's Gaussian policy
                initializes near 0, so ~half of early samples are clipped to 0 and
                wasted. Fix: pass policy_kwargs={"squash_output": True} in
                scripts/train.py, which applies tanh + rescale to guarantee every
                sampled action lands in [0, max_bid_multiplier] from step one.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        self.env = env

        self.policy = config.get("policy", "MlpPolicy")
        self.verbose = int(config.get("verbose", 1))
        self.device = config.get("device", "auto")
        self.tensorboard_log = config.get("tensorboard_log", None)

        # Core PPO hyperparameters.
        # n_steps=384 = 48 ticks/episode × 8 envs — keeps GAE within complete episodes.
        self.learning_rate = float(config.get("lr", 3e-4))
        self.n_steps = int(config.get("n_steps", 384))
        self.batch_size = int(config.get("batch_size", 128))
        self.n_epochs = int(config.get("n_epochs", 10))
        self.gamma = float(config.get("gamma", 0.99))
        self.gae_lambda = float(config.get("gae_lambda", 0.95))
        self.clip_range = float(config.get("clip_range", 0.2))
        # ent_coef=0.01: exploration bonus needed for continuous [0, max_bid_multiplier]
        # action space; 0.0 causes the policy to collapse to a narrow region early.
        self.ent_coef = float(config.get("ent_coef", 0.01))
        self.vf_coef = float(config.get("vf_coef", 0.5))
        self.max_grad_norm = float(config.get("max_grad_norm", 0.5))
        self.seed = config.get("seed", None)

        self.model: Optional[PPO] = None
        if env is not None:
            self.model = self._build_model(env)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self, env) -> PPO:
        """Construct and return a fresh SB3 PPO instance for the given env."""
        return PPO(
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_env(self, env) -> None:
        """Attach or replace the training environment.

        If no model exists yet, builds one. Otherwise delegates to SB3's set_env.
        """
        self.env = env
        if self.model is None:
            self.model = self._build_model(env)
        else:
            self.model.set_env(env)

    def select_action(self, observation, deterministic: bool = False):
        """Predict an action from the current policy.

        Args:
            observation (np.ndarray): Current observation.
            deterministic (bool): Whether to use deterministic policy output.

        Returns:
            action (np.ndarray): Predicted action (bid multiplier).
            None: log_prob — not exposed by SB3's public predict() API.
            None: value   — not exposed by SB3's public predict() API.
        """
        if self.model is None:
            raise ValueError("PPO model is not initialized. Call set_env(env) first.")
        obs = np.asarray(observation, dtype=np.float32)
        action, _states = self.model.predict(obs, deterministic=deterministic)
        return action, None, None

    def update(self, rollout_buffer=None, total_timesteps: Optional[int] = None, **learn_kwargs):
        """Train the SB3 PPO model via model.learn().

        `rollout_buffer` is ignored — SB3 manages its own rollout buffer internally.

        Args:
            rollout_buffer: Ignored. Kept only for interface compatibility.
            total_timesteps (int): Number of environment steps to train for.
                Defaults to config['total_timesteps'] or 500_000.
            **learn_kwargs: Extra keyword arguments forwarded to model.learn().

        Returns:
            dict: {'trained_timesteps': int}
        """
        if self.model is None:
            raise ValueError("PPO model is not initialized. Call set_env(env) first.")
        if total_timesteps is None:
            total_timesteps = int(self.config.get("total_timesteps", 500_000))
        self.model.learn(total_timesteps=total_timesteps, **learn_kwargs)
        return {"trained_timesteps": total_timesteps}

    def evaluate(self, env, n_eval_episodes: int = 10) -> dict:
        """Run deterministic rollouts and return RTB-specific evaluation metrics.

        This is distinct from SB3's internal EvalCallback, which only tracks
        ep_rew_mean. This method computes the domain metrics needed to compare
        agents in the research paper: ROI, budget utilization, and win rate.

        Args:
            env: A single (non-vectorized) AuctionNetGymEnv instance.
                Must not be wrapped with VecNormalize — pass the raw env so
                that info dict values are in original units.
            n_eval_episodes (int): Number of complete episodes to roll out.

        Returns:
            dict with keys:
                roi (float): total conversion value / total cost.
                    Higher is better. 0.0 if agent spent nothing.
                budget_utilization (float): total cost / (n_eval_episodes * budget).
                    Target ~0.90–1.0.
                win_rate (float): total auctions won / total auctions entered.
        """
        if self.model is None:
            raise ValueError("PPO model is not initialized. Call set_env(env) first.")

        total_conversion_value = 0.0
        total_cost = 0.0
        total_won = 0
        total_auctions = 0
        budget = env.budget

        for _ in range(n_eval_episodes):
            obs, _ = env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                action, _, _ = self.select_action(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = env.step(action)
                total_conversion_value += info["conversion_value"]
                total_cost += info["cost"]
                total_won += info["won"]
                total_auctions += info.get("total_pvs", env._pv_gen.pv_values[0].shape[0])

        roi = total_conversion_value / total_cost if total_cost > 0 else 0.0
        budget_utilization = total_cost / (n_eval_episodes * budget)
        win_rate = total_won / total_auctions if total_auctions > 0 else 0.0

        return {
            "roi": roi,
            "budget_utilization": budget_utilization,
            "win_rate": win_rate,
        }

    def save(self, path: str):
        """Save the SB3 model checkpoint to disk.

        If the training env is a VecNormalize wrapper, its running statistics
        (observation mean/var, reward mean/var) are saved as a sidecar file at
        '<path>_vecnormalize.pkl'. Both files must be present to restore a
        correctly functioning policy at inference time.

        Args:
            path (str): Save path without extension, e.g. 'saved_models/ppo_run1'.
                SB3 appends '.zip' to the model file automatically.
        """
        if self.model is None:
            raise ValueError("PPO model is not initialized. Nothing to save.")
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
        if isinstance(self.env, VecNormalize):
            self.env.save(str(save_path) + "_vecnormalize.pkl")

    def load(self, path: str, env=None):
        """Load a saved SB3 PPO checkpoint.

        If a VecNormalize sidecar file ('<path>_vecnormalize.pkl') exists and
        the provided env is a VecNormalize instance, the saved running statistics
        are restored into it before attaching it to the model. This ensures
        observations are scaled identically to how they were during training.

        Args:
            path (str): Path to the checkpoint (with or without '.zip').
            env: Optional environment to attach after loading. Pass a
                VecNormalize-wrapped env to trigger stats restoration.

        Returns:
            PPOAgent: self
        """
        if env is not None:
            self.env = env

        # Strip .zip so the sidecar path is consistent with save() regardless of
        # whether the caller passed "best_model" or "best_model.zip".
        p = Path(path)
        base_path = str(p.with_suffix("")) if p.suffix == ".zip" else str(p)
        vecnorm_path = base_path + "_vecnormalize.pkl"
        if isinstance(self.env, VecNormalize) and Path(vecnorm_path).exists():
            self.env = VecNormalize.load(vecnorm_path, self.env.venv)
            self.env.training = False
            self.env.norm_reward = False

        self.model = PPO.load(path, env=self.env, device=self.device)
        return self
