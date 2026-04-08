"""
Gymnasium-compliant wrapper around AuctionNet's simul_bidding_env.

AuctionNet (NeurIPS 2024, Alibaba) provides a tick-based simulation of
real-time ad auctions with 48 competing advertisers. This module wraps that
environment into the standard Gymnasium API so that SB3's PPO (or any
Gymnasium-compatible algorithm) can train on it directly.

MDP mapping:
  - 1 Gym step  = 1 AuctionNet tick  (48 steps per episode)
  - Action      = scalar bid multiplier α ∈ [0, max_bid_multiplier]
  - Bids        = α × pValues  (computed internally for all PVs in the tick)
  - Reward      = conversion_value_won - cost  (simple baseline, no λ yet)
  - Observation = 7-dim vector (see _get_obs docstring)

Usage:
    import yaml
    from environment.gym_wrapper import AuctionNetGymEnv

    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)["environment"]

    env = AuctionNetGymEnv(config)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import math
import os
import sys

import gymnasium
import numpy as np
from gymnasium.spaces import Box

# Add AuctionNet sibling repo to path.
# auctionnet_path in config is relative to the project root (one level above environment/).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)


def _resolve_auctionnet_path(config_path: str) -> str:
    """Resolve auctionnet_path from config relative to the project root."""
    return os.path.normpath(os.path.join(_PROJECT_ROOT, config_path))


# ---------------------------------------------------------------------------
# Budget / CPA / category constants mirrored from AuctionNet's Controller.
# Kept here so we don't need to instantiate the full Controller.
# ---------------------------------------------------------------------------

_BUDGETS = [
    2900, 4350, 3000, 2400, 4800, 2000, 2050, 3500,
    4600, 2000, 2800, 2350, 2050, 2900, 4750, 3450,
    2000, 3500, 2200, 2700, 3100, 2100, 4850, 4100,
    2000, 4800, 3050, 4250, 2850, 2250, 2000, 3900,
    2000, 3250, 4450, 3550, 2700, 2100, 4650, 2000,
    3400, 2650, 2300, 4100, 4800, 4450, 2000, 2050,
]

_CPAS = [
    100, 70, 90, 110, 60, 130, 120, 80,
    70, 130, 100, 110, 120, 90, 60, 80,
    130, 80, 110, 100, 90, 120, 60, 70,
    120, 60, 90, 70, 100, 110, 130, 80,
    120, 90, 70, 80, 100, 110, 60, 130,
    90, 100, 110, 80, 60, 70, 130, 120,
]

_NUM_AGENT_CATEGORY = 8   # 48 agents / 6 categories = 8 agents per category
_NUM_CATEGORY = 6         # 6 industry categories

# category[i] = i // num_agent_category  (from Controller)
_CATEGORIES = [i // _NUM_AGENT_CATEGORY for i in range(48)]


def _build_competitors(player_index: int):
    """
    Instantiate 47 PID rule-based competitor agents.
    PID is fully self-contained — no CSV or model files required.

    Returns a list of 48 agent slots; slot player_index is None (filled by
    the wrapper with the player's bid multiplier logic).
    """
    from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy

    agents = []
    for i in range(48):
        if i == player_index:
            agents.append(None)  # placeholder for the player
            continue
        agent = PidBiddingStrategy(exp_tempral_ratio=np.ones(48))
        agent.budget = _BUDGETS[i]
        agent.cpa = _CPAS[i]
        agent.category = _CATEGORIES[i]
        agent.name = f"agent_{i}"
        agents.append(agent)
    return agents


def _adjust_over_cost(bids: np.ndarray, over_cost_ratio: np.ndarray,
                      slot_coefficients: np.ndarray, winner_pit: np.ndarray) -> None:
    """Drop random winning bids for agents that would exceed their budget.

    Mirrors adjust_over_cost from AuctionNet's run_test.py.
    """
    overcost_indices = np.where(over_cost_ratio > 0)[0]
    rng = np.random.default_rng(seed=1)
    for agent_idx in overcost_indices:
        for slot_pos, _ in enumerate(slot_coefficients):
            winner_indices = winner_pit[:, slot_pos]
            pv_indices = np.where(winner_indices == agent_idx)[0]
            num_to_drop = math.ceil(pv_indices.size * over_cost_ratio[agent_idx])
            if num_to_drop > 0:
                dropped = rng.choice(pv_indices, num_to_drop, replace=False)
                bids[dropped, agent_idx] = 0


def _get_winner(slot_pit: np.ndarray) -> np.ndarray:
    """Return winner matrix (n_pv, 3) from slot assignment matrix.

    Mirrors get_winner from AuctionNet's run_test.py.
    """
    slot_pit_t = slot_pit.T  # (n_pv, n_agent)
    num_pv = slot_pit_t.shape[0]
    winner = np.full((num_pv, 3), -1, dtype=int)
    for pos in range(1, 4):
        winning = np.argwhere(slot_pit_t == pos)
        if winning.size > 0:
            pv_idx, agent_idx = winning.T
            winner[pv_idx, pos - 1] = agent_idx
    return winner


class AuctionNetGymEnv(gymnasium.Env):
    """Gymnasium wrapper around AuctionNet's simul_bidding_env.

    Each episode runs for num_ticks=48 steps. At each step the agent supplies
    a scalar bid multiplier α; the wrapper bids α × pValue for every PV
    opportunity in that tick, runs the second-price auction against 47
    rule-based competitors, and returns the aggregate outcome.

    Observation space (7-dim Box, float32):
        [0] remaining_budget / initial_budget
        [1] tick_index / num_ticks
        [2] mean pValue for current tick's PV opportunities
        [3] std pValue for current tick's PV opportunities
        [4] least_winning_cost from previous tick / initial_budget
        [5] win rate over the last 3 ticks (0 if tick < 1)
        [6] pacing_ratio = spend_fraction / time_fraction (1.0 at tick 0)

    Action space (1-dim Box, float32):
        α ∈ [0, max_bid_multiplier]  →  bids = α × pValues

    Reward:
        r_t = conversion_value_won_t - cost_t

    Termination:
        remaining_budget < min_remaining_budget  OR  tick == num_ticks - 1
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict, render_mode: str = None):
        """Initialize the environment.

        Args:
            config (dict): The 'environment' section of configs/default.yaml.
                Required keys: budget, num_ticks, num_episode, player_index,
                max_bid_multiplier, auctionnet_path, pv_num, min_remaining_budget.
            render_mode: Unused; kept for Gymnasium API compatibility.
        """
        super().__init__()

        # --- resolve and inject AuctionNet path ---
        auctionnet_abs = _resolve_auctionnet_path(config["auctionnet_path"])
        if auctionnet_abs not in sys.path:
            sys.path.insert(0, auctionnet_abs)

        # --- store config ---
        self.budget = float(config["budget"])
        self.num_ticks = int(config["num_ticks"])
        self.num_episode = int(config["num_episode"])
        self.player_index = int(config["player_index"])
        self.max_bid_multiplier = float(config["max_bid_multiplier"])
        self.pv_num = int(config["pv_num"])
        self.min_remaining_budget = float(config["min_remaining_budget"])

        # --- import AuctionNet modules (after path is set) ---
        from simul_bidding_env.Environment.BiddingEnv import BiddingEnv
        from simul_bidding_env.PvGenerator.NeurIPSPvGen import NeurIPSPvGen

        self._BiddingEnv = BiddingEnv
        self._NeurIPSPvGen = NeurIPSPvGen

        # --- instantiate auction core ---
        self._bidding_env = BiddingEnv()
        self._pv_gen = NeurIPSPvGen(
            episode=0,
            num_tick=self.num_ticks,
            num_agent=48,
            num_agent_category=_NUM_AGENT_CATEGORY,
            num_category=_NUM_CATEGORY,
            pv_num=self.pv_num,
        )

        # --- build competitor agents (rule-based only, no model weights) ---
        self._agents = _build_competitors(self.player_index)

        # --- Gymnasium spaces ---
        self.observation_space = Box(
            low=np.zeros(7, dtype=np.float32),
            high=np.full(7, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([self.max_bid_multiplier], dtype=np.float32),
            dtype=np.float32,
        )

        # --- internal state (initialized in reset) ---
        self._tick = 0
        self._episode = 0
        self._remaining_budget = self.budget
        self._total_spend = 0.0
        self._last_lwc = 0.0          # least winning cost from previous tick
        self._recent_xi = []          # list of (n_won, n_total) per tick for win rate
        self._current_pv_values = None
        self._current_pvalue_sigmas = None

        # history buffers (same structure as run_test.py)
        self._hist_pvalue_infos = []
        self._hist_bids = []
        self._hist_auction_results = []
        self._hist_impression_results = []
        self._hist_least_winning_costs = []

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Reset to a new episode.

        Returns:
            observation (np.ndarray): shape (7,)
            info (dict): empty dict
        """
        super().reset(seed=seed)

        # pick episode (random if seeded, else sequential)
        self._episode = int(self.np_random.integers(0, self.num_episode))

        # reset AuctionNet internals
        # Note: NeurIPSPvGen.reset() drops all constructor args back to defaults,
        # so we reinstantiate it directly to preserve pv_num and other config.
        self._bidding_env.reset(episode=self._episode)
        self._pv_gen = self._NeurIPSPvGen(
            episode=self._episode,
            num_tick=self.num_ticks,
            num_agent=48,
            num_agent_category=_NUM_AGENT_CATEGORY,
            num_category=_NUM_CATEGORY,
            pv_num=self.pv_num,
        )

        # reset competitors
        for i, agent in enumerate(self._agents):
            if agent is None:
                continue
            agent.reset()
            agent.remaining_budget = _BUDGETS[i]

        # reset player state
        self._tick = 0
        self._remaining_budget = self.budget
        self._total_spend = 0.0
        self._last_lwc = 0.0
        self._recent_xi = []

        # reset history buffers
        self._hist_pvalue_infos = []
        self._hist_bids = []
        self._hist_auction_results = []
        self._hist_impression_results = []
        self._hist_least_winning_costs = []

        # pre-fetch tick 0 PV data so _get_obs can read mean/std pValues
        self._current_pv_values = self._pv_gen.pv_values[0]
        self._current_pvalue_sigmas = self._pv_gen.pValueSigmas[0]

        return self._get_obs(), {}

    def step(self, action):
        """Run one tick of the auction simulation.

        Args:
            action (np.ndarray): shape (1,), bid multiplier α.

        Returns:
            observation (np.ndarray): shape (7,)
            reward (float)
            terminated (bool)
            truncated (bool)
            info (dict)
        """
        alpha = float(np.clip(action, 0.0, self.max_bid_multiplier))

        pv_values = self._pv_gen.pv_values[self._tick]         # (n_pv, 48)
        pvalue_sigmas = self._pv_gen.pValueSigmas[self._tick]  # (n_pv, 48)
        p = self.player_index

        # --- collect bids from all 48 agents ---
        remaining_budgets = np.array([
            self._remaining_budget if i == p else (
                agent.remaining_budget if agent is not None else 0.0
            )
            for i, agent in enumerate(self._agents)
        ])

        all_bids = []
        for i, agent in enumerate(self._agents):
            if i == p:
                bids_i = alpha * pv_values[:, i]
            elif agent is None or remaining_budgets[i] < self._bidding_env.min_remaining_budget:
                bids_i = np.zeros(pv_values.shape[0])
            else:
                hist_pval = [x[i] for x in self._hist_pvalue_infos]
                hist_bid = [x[i] for x in self._hist_bids]
                hist_auction = [x[i] for x in self._hist_auction_results]
                hist_impression = [x[i] for x in self._hist_impression_results]
                bids_i = agent.bidding(
                    self._tick,
                    pv_values[:, i],
                    pvalue_sigmas[:, i],
                    hist_pval,
                    hist_bid,
                    hist_auction,
                    hist_impression,
                    self._hist_least_winning_costs,
                )
            all_bids.append(np.asarray(bids_i, dtype=np.float64))

        bids = np.array(all_bids).T  # (n_pv, 48)
        bids = np.clip(bids, 0, None)

        # --- overcost-adjustment loop (mirrors run_test.py) ---
        winner_pit = None
        while True:
            xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit, \
                lwc_pit, market_price_pit = \
                self._bidding_env.simulate_ad_bidding(pv_values, pvalue_sigmas, bids)

            real_cost = (cost_pit * is_exposed_pit).sum(axis=1)  # (48,)
            over_cost_ratio = np.maximum(
                (real_cost - remaining_budgets) / (real_cost + 1e-4), 0
            )
            winner_pit = _get_winner(slot_pit)
            if over_cost_ratio.max() == 0:
                break
            _adjust_over_cost(bids, over_cost_ratio,
                               self._bidding_env.slot_coefficients, winner_pit)

        # --- update budgets ---
        self._remaining_budget -= real_cost[p]
        self._total_spend += real_cost[p]
        for i, agent in enumerate(self._agents):
            if agent is not None and i != p:
                agent.remaining_budget -= real_cost[i]

        # --- compute reward: conversion value - cost (player only) ---
        player_conversion_value = float(conversion_action_pit[p].sum())
        player_cost = float(real_cost[p])
        reward = player_conversion_value - player_cost

        # --- track least winning cost and win rate ---
        self._last_lwc = float(lwc_pit.mean())  # average market clearing price
        n_won = int(xi_pit[p].sum())
        n_total = int(xi_pit.shape[1])
        self._recent_xi.append((n_won, n_total))
        if len(self._recent_xi) > 3:
            self._recent_xi.pop(0)

        # --- update history buffers ---
        pvalue_info = np.stack((pv_values.T, pvalue_sigmas.T), axis=-1)  # (48, n_pv, 2)
        self._hist_pvalue_infos.append(pvalue_info)
        self._hist_bids.append(bids.T)                                    # (48, n_pv)
        auction_info = np.stack((xi_pit, slot_pit, cost_pit), axis=-1)
        self._hist_auction_results.append(auction_info)
        impression_info = np.stack((is_exposed_pit, conversion_action_pit), axis=-1)
        self._hist_impression_results.append(impression_info)
        self._hist_least_winning_costs.append(lwc_pit)

        # --- advance tick ---
        self._tick += 1

        # --- check termination ---
        terminated = (
            self._remaining_budget < self.min_remaining_budget
            or self._tick >= self.num_ticks
        )
        truncated = False

        # --- fetch next tick PV data for observation (if episode continues) ---
        if not terminated:
            self._current_pv_values = self._pv_gen.pv_values[self._tick]
            self._current_pvalue_sigmas = self._pv_gen.pValueSigmas[self._tick]
        else:
            self._current_pv_values = pv_values   # reuse last tick's values

        info = {
            "won": n_won,
            "cost": player_cost,
            "conversion_value": player_conversion_value,
            "tick": self._tick,
            "total_budget": self.budget,
            "remaining_budget": self._remaining_budget,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build the 7-dimensional observation vector.

        Returns:
            np.ndarray: shape (7,), dtype float32, all values >= 0.
                [0] remaining_budget / initial_budget        (budget fraction)
                [1] tick_index / num_ticks                   (time progress)
                [2] mean pValue this tick                    (opportunity quality)
                [3] std pValue this tick                     (opportunity diversity)
                [4] last least_winning_cost / initial_budget (market price proxy)
                [5] recent win rate (last 3 ticks)
                [6] pacing_ratio = spend_frac / time_frac   (>1 = overspending)
        """
        p = self.player_index
        pv_col = self._current_pv_values[:, p]

        budget_frac = self._remaining_budget / self.budget
        time_frac = self._tick / self.num_ticks
        mean_pval = float(pv_col.mean())
        std_pval = float(pv_col.std())
        norm_price = self._last_lwc / self.budget

        if self._recent_xi:
            total_won = sum(w for w, _ in self._recent_xi)
            total_pv = sum(t for _, t in self._recent_xi)
            win_rate = total_won / total_pv if total_pv > 0 else 0.0
        else:
            win_rate = 0.0

        spend_frac = self._total_spend / self.budget
        pacing_ratio = spend_frac / time_frac if time_frac > 0 else 1.0

        return np.array(
            [budget_frac, time_frac, mean_pval, std_pval,
             norm_price, win_rate, pacing_ratio],
            dtype=np.float32,
        )
