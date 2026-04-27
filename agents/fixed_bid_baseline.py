"""
Fixed-bid baseline agent for real-time ad bidding.

The simplest possible bidding strategy: always submit the same fixed bid
regardless of context, budget remaining, or market conditions. Serves as a
lower-bound baseline that any learned policy should comfortably beat.

This version is intentionally aligned with PPOAgent's project-facing interface:

    agent = FixedBidBaseline(env.observation_space, env.action_space, config, env=env)
    action, log_prob, value = agent.select_action(obs)
    agent.update(total_timesteps=10_000)   # no-op
    agent.save("saved_models/fixed_bid.json")
    agent.load("saved_models/fixed_bid.json")

Unlike PPOAgent, this baseline does not learn. `update()` is a no-op and only
exists so the same training/evaluation code can call both agents uniformly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


class FixedBidBaseline:
    """Baseline agent that always bids a constant fixed amount.

    No learning occurs. The bid amount is set at construction time and remains
    constant throughout the episode. Useful as a sanity-check lower bound: a
    learned PPO agent should generally outperform this on ROI/reward.

    Args:
        observation_space: Gymnasium observation space. Kept for interface
            consistency with PPOAgent.
        action_space: Gymnasium action space. Used to infer action shape and
            optionally clip the fixed bid to valid bounds.
        config (dict): Baseline config. Expected key: ``fixed_bid``.
        env: Optional Gymnasium-compatible environment. Stored only for
            interface consistency.

    Attributes:
        fixed_bid (float): Constant bid value submitted at every step.
    """

    def __init__(
        self,
        observation_space=None,
        action_space=None,
        config: Optional[dict[str, Any]] = None,
        env=None,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config or {}
        self.env = env

        if "fixed_bid" not in self.config:
            raise ValueError(
                "FixedBidBaseline requires config['fixed_bid'], e.g. "
                "{'fixed_bid': 0.5}."
            )

        self.fixed_bid = float(self.config["fixed_bid"])
        self.clip_action = bool(self.config.get("clip_action", True))

    def set_env(self, env) -> None:
        """Attach or replace the environment, matching PPOAgent's interface."""
        self.env = env
        if hasattr(env, "observation_space"):
            self.observation_space = env.observation_space
        if hasattr(env, "action_space"):
            self.action_space = env.action_space

    def _format_action(self):
        """Return fixed bid in a shape compatible with the action space."""
        action = self.fixed_bid

        # If the environment uses a continuous Box action space, PPO/SB3 usually
        # returns a NumPy array with shape == action_space.shape. Match that.
        if self.action_space is not None and hasattr(self.action_space, "shape"):
            shape = self.action_space.shape
            if shape is not None and len(shape) > 0:
                action = np.full(shape, self.fixed_bid, dtype=np.float32)
            else:
                action = np.asarray(self.fixed_bid, dtype=np.float32)
        else:
            action = np.asarray(self.fixed_bid, dtype=np.float32)

        if self.clip_action and self.action_space is not None:
            low = getattr(self.action_space, "low", None)
            high = getattr(self.action_space, "high", None)
            if low is not None and high is not None:
                action = np.clip(action, low, high).astype(np.float32)

        return action

    def select_action(self, observation, deterministic: bool = True):
        """Return the fixed bid, ignoring the observation entirely.

        Args:
            observation: Current environment observation. Ignored.
            deterministic (bool): Kept for compatibility with PPOAgent.

        Returns:
            tuple: ``(action, None, None)``, matching PPOAgent.select_action().
                ``log_prob`` and ``value`` are unavailable for this rule-based
                baseline, so both are returned as None.
        """
        action = self._format_action()
        return action, None, None

    def predict(self, observation, deterministic: bool = True):
        """SB3-like predict method for evaluation code that calls model.predict().

        Returns:
            tuple: ``(action, None)`` matching Stable-Baselines3 predict style.
        """
        action, _, _ = self.select_action(observation, deterministic=deterministic)
        return action, None

    def update(self, rollout_buffer=None, total_timesteps: Optional[int] = None, **learn_kwargs):
        """No-op training method, matching PPOAgent.update().

        Fixed-bid baselines do not learn. This method exists only so shared
        scripts can call ``agent.update(...)`` without special-casing baselines.

        Returns:
            dict: Minimal diagnostics.
        """
        return {
            "trained_timesteps": 0,
            "requested_timesteps": total_timesteps,
            "fixed_bid": self.fixed_bid,
            "note": "FixedBidBaseline does not learn; update() is a no-op.",
        }

    def save(self, path: str):
        """Serialize the fixed bid config to disk as JSON.

        Args:
            path (str): Save path, e.g. ``saved_models/fixed_bid.json``.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "fixed_bid": self.fixed_bid,
            "clip_action": self.clip_action,
        }
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load(self, path: str, env=None):
        """Load a saved fixed bid config from JSON.

        Args:
            path (str): Path to saved JSON config.
            env: Optional env to attach after loading.

        Returns:
            FixedBidBaseline: self
        """
        load_path = Path(path)
        with load_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if "fixed_bid" not in payload:
            raise ValueError(f"Saved baseline config at {path} missing 'fixed_bid'.")

        self.fixed_bid = float(payload["fixed_bid"])
        self.clip_action = bool(payload.get("clip_action", True))
        self.config.update(payload)

        if env is not None:
            self.set_env(env)

        return self
