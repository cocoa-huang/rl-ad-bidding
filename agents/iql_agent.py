"""
Implicit Q-Learning (IQL) agent for real-time ad bidding.

This module implements a minimal project-friendly IQL agent with the same
high-level interface as PPOAgent:

    agent = IQLAgent(observation_space, action_space, agent_config, env=env)
    agent.update(offline_dataset=dataset, total_timesteps=10_000)
    action, _, _ = agent.select_action(obs)
    agent.save("saved_models/iql_agent")
    agent.load("saved_models/iql_agent")

Important:
    IQL is an offline RL algorithm. Unlike PPO, it does not learn from online
    environment rollout during update(). It expects an offline transition
    dataset containing observations, actions, rewards, next_observations, and
    dones. For now, this implementation is intentionally minimal so it can be
    used as a baseline and integrated into the existing train/evaluate pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    """Simple fully connected network."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IQLAgent:
    """Minimal Implicit Q-Learning agent with PPOAgent-compatible interface."""

    def __init__(
        self,
        observation_space,
        action_space,
        config: dict,
        env=None,
    ):
        """
        Initialize the IQL agent.

        Args:
            observation_space: Gymnasium observation space.
            action_space: Gymnasium action space. Currently assumes continuous Box action.
            config (dict): Hyperparameters from configs/*.yaml under 'agent'.
            env: Optional Gymnasium-compatible environment or VecEnv. Kept for
                interface consistency with PPOAgent.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        self.env = env

        device_cfg = config.get("device", "auto")
        if device_cfg == "auto":
            device_cfg = "cuda" if torch.cuda.is_available() else "cpu" 
            

        self.device = torch.device(device_cfg)
        self.verbose = int(config.get("verbose", 1))
        self.seed = config.get("seed", None)
        if self.seed is not None:
            np.random.seed(int(self.seed))
            torch.manual_seed(int(self.seed))

        self.gamma = float(config.get("gamma", 0.99))
        self.tau = float(config.get("iql_tau", 0.7))
        self.beta = float(config.get("iql_beta", 3.0))
        self.expectile = self.tau  # alias for clarity
        self.max_advantage_weight = float(config.get("max_advantage_weight", 100.0))

        self.learning_rate = float(config.get("lr", 3e-4))
        self.batch_size = int(config.get("batch_size", 256))
        self.n_epochs = int(config.get("n_epochs", 10))
        self.hidden_dims = tuple(config.get("hidden_dims", [256, 256]))

        self.obs_dim = int(np.prod(observation_space.shape))
        self.action_dim = int(np.prod(action_space.shape))

        self.action_low = torch.as_tensor(action_space.low, dtype=torch.float32, device=self.device).view(1, -1)
        self.action_high = torch.as_tensor(action_space.high, dtype=torch.float32, device=self.device).view(1, -1)

        # Networks:
        # V(s): scalar state value
        # Q(s,a): scalar action value
        # pi(s): deterministic behavior-improvement policy for continuous action
        self.value_net = MLP(self.obs_dim, 1, self.hidden_dims).to(self.device)
        self.q_net = MLP(self.obs_dim + self.action_dim, 1, self.hidden_dims).to(self.device)
        self.policy_net = MLP(self.obs_dim, self.action_dim, self.hidden_dims).to(self.device)

        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def set_env(self, env) -> None:
        """Attach or replace the environment. Kept for PPOAgent interface consistency."""
        self.env = env

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _flatten_obs(self, observation) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        else:
            obs = obs.reshape(obs.shape[0], -1)
        return obs

    def _scale_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Map tanh output from [-1, 1] to the environment action range."""
        squashed = torch.tanh(raw_action)
        return self.action_low + 0.5 * (squashed + 1.0) * (self.action_high - self.action_low)

    def select_action(self, observation, deterministic: bool = False):
        """
        Predict an action from the current IQL policy.

        Args:
            observation (np.ndarray): Current observation.
            deterministic (bool): Kept for interface compatibility. This minimal
                implementation uses a deterministic policy regardless.

        Returns:
            action (np.ndarray): Predicted action.
            log_prob: None. Not exposed by this minimal deterministic policy.
            value: None. Kept for PPOAgent-compatible return signature.
        """
        obs = self._flatten_obs(observation)
        obs_tensor = self._to_tensor(obs)

        self.policy_net.eval()
        with torch.no_grad():
            raw_action = self.policy_net(obs_tensor)
            action = self._scale_action(raw_action)
        self.policy_net.train()

        action_np = action.cpu().numpy().astype(np.float32)

        # Match SB3/DummyVecEnv convention: if input is single obs, return one action.
        if np.asarray(observation).ndim == 1:
            action_np = action_np.reshape(self.action_space.shape)

        return action_np, None, None

    def _prepare_dataset(self, offline_dataset: Any) -> TensorDataset:
        """
        Convert an offline dataset into a TensorDataset.

        Expected dictionary keys:
            observations or obs
            actions
            rewards
            next_observations or next_obs
            dones
        """
        if offline_dataset is None:
            raise ValueError(
                "IQL requires an offline dataset. Pass offline_dataset to update(), "
                "or modify train.py to load a replay dataset before calling update()."
            )

        if not isinstance(offline_dataset, dict):
            raise TypeError("offline_dataset must be a dict of numpy arrays or tensors.")

        obs = offline_dataset.get("observations", offline_dataset.get("obs"))
        actions = offline_dataset.get("actions")
        rewards = offline_dataset.get("rewards")
        next_obs = offline_dataset.get("next_observations", offline_dataset.get("next_obs"))
        dones = offline_dataset.get("dones")

        missing = []
        for name, value in [
            ("observations/obs", obs),
            ("actions", actions),
            ("rewards", rewards),
            ("next_observations/next_obs", next_obs),
            ("dones", dones),
        ]:
            if value is None:
                missing.append(name)
        if missing:
            raise KeyError(f"offline_dataset is missing required key(s): {missing}")

        obs = np.asarray(obs, dtype=np.float32).reshape(len(obs), -1)
        actions = np.asarray(actions, dtype=np.float32).reshape(len(actions), -1)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(len(next_obs), -1)
        dones = np.asarray(dones, dtype=np.float32).reshape(-1, 1)

        return TensorDataset(
            self._to_tensor(obs),
            self._to_tensor(actions),
            self._to_tensor(rewards),
            self._to_tensor(next_obs),
            self._to_tensor(dones),
        )

    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        """Expectile regression loss used by IQL value fitting."""
        weight = torch.where(diff > 0, self.expectile, 1.0 - self.expectile)
        return (weight * diff.pow(2)).mean()

    def update(self, rollout_buffer=None, total_timesteps: Optional[int] = None, **learn_kwargs):
        """
        Train IQL on an offline transition dataset.

        Args:
            rollout_buffer: Ignored unless used as offline_dataset fallback.
            total_timesteps (int): Approximate gradient-step budget. If provided,
                it overrides n_epochs by training for this many mini-batch updates.
            **learn_kwargs: Should include offline_dataset=dict(...). For backward
                compatibility, rollout_buffer can also be that dict.

        Returns:
            dict: Minimal diagnostics.
        """
        offline_dataset = learn_kwargs.pop("offline_dataset", None)
        if offline_dataset is None and isinstance(rollout_buffer, dict):
            offline_dataset = rollout_buffer

        dataset = self._prepare_dataset(offline_dataset)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        if total_timesteps is None:
            max_updates = None
            n_epochs = self.n_epochs
        else:
            max_updates = int(total_timesteps)
            n_epochs = 10**9  # stopped by max_updates

        update_count = 0
        last_losses = {"value_loss": np.nan, "q_loss": np.nan, "policy_loss": np.nan}

        for _epoch in range(n_epochs):
            for obs, actions, rewards, next_obs, dones in loader:
                with torch.no_grad():
                    next_v = self.value_net(next_obs)
                    q_target = rewards + self.gamma * (1.0 - dones) * next_v

                # Q update: regress Q(s,a) to r + gamma V(s')
                q_input = torch.cat([obs, actions], dim=-1)
                q_pred = self.q_net(q_input)
                q_loss = F.mse_loss(q_pred, q_target)
                self.q_optimizer.zero_grad()
                q_loss.backward()
                self.q_optimizer.step()

                # V update: expectile regression against current Q(s,a)
                with torch.no_grad():
                    q_detached = self.q_net(q_input)
                v_pred = self.value_net(obs)
                value_loss = self._expectile_loss(q_detached - v_pred)
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

                # Policy update: advantage-weighted behavioral cloning
                with torch.no_grad():
                    q_for_adv = self.q_net(q_input)
                    v_for_adv = self.value_net(obs)
                    advantage = q_for_adv - v_for_adv
                    weights = torch.exp(self.beta * advantage).clamp(max=self.max_advantage_weight)

                pred_actions = self._scale_action(self.policy_net(obs))
                bc_loss = ((pred_actions - actions).pow(2).sum(dim=-1, keepdim=True))
                policy_loss = (weights * bc_loss).mean()
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                update_count += 1
                last_losses = {
                    "value_loss": float(value_loss.detach().cpu().item()),
                    "q_loss": float(q_loss.detach().cpu().item()),
                    "policy_loss": float(policy_loss.detach().cpu().item()),
                }

                if max_updates is not None and update_count >= max_updates:
                    return {"trained_updates": update_count, **last_losses}

        return {"trained_updates": update_count, **last_losses}

    def save(self, path: str):
        """
        Save IQL model checkpoint.

        Args:
            path (str): Save path, e.g. 'saved_models/iql_agent'. A '.pt'
                suffix is added if no suffix is provided.
        """
        save_path = Path(path)
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".pt")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "config": self.config,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "value_net": self.value_net.state_dict(),
            "q_net": self.q_net.state_dict(),
            "policy_net": self.policy_net.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
        }
        torch.save(checkpoint, save_path)

    def load(self, path: str, env=None):
        """
        Load a saved IQL checkpoint.

        Args:
            path (str): Path to checkpoint. If no suffix is provided, '.pt' is tried.
            env: Optional env to attach after loading.

        Returns:
            IQLAgent: self
        """
        if env is not None:
            self.env = env

        load_path = Path(path)
        if not load_path.exists() and load_path.suffix == "":
            pt_path = load_path.with_suffix(".pt")
            if pt_path.exists():
                load_path = pt_path

        checkpoint = torch.load(load_path, map_location=self.device)
        self.value_net.load_state_dict(checkpoint["value_net"])
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.policy_net.load_state_dict(checkpoint["policy_net"])

        if "value_optimizer" in checkpoint:
            self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        if "q_optimizer" in checkpoint:
            self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        if "policy_optimizer" in checkpoint:
            self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])

        return self
