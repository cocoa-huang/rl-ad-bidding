"""
Common simulator evaluator for PPO, AuctionNet offline RL, and fixed-alpha policies.

This script exists to answer the project-level question fairly:
can PPO outperform AuctionNet's offline RL baseline under the same simulation
conditions?

It uses one AuctionNet simulation loop for every policy and computes metrics
from the same raw arrays:
  - conversions
  - cost
  - budget utilization
  - aggregate CPA
  - conversion ROI
  - slot win rate
  - exposure rate
  - shared shaped reward

Usage:
    python scripts/common_policy_eval.py --config configs/gcp-run-9-selective.yaml \
        --run-name gcp-run-9-selective --episodes 48 --fixed-alphas 50 100 130 150
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NUM_AGENT_CATEGORY = 8
NUM_CATEGORY = 6
NUM_AGENTS = 48

DEFAULT_BUDGETS = np.array([
    2900, 4350, 3000, 2400, 4800, 2000, 2050, 3500,
    4600, 2000, 2800, 2350, 2050, 2900, 4750, 3450,
    2000, 3500, 2200, 2700, 3100, 2100, 4850, 4100,
    2000, 4800, 3050, 4250, 2850, 2250, 2000, 3900,
    2000, 3250, 4450, 3550, 2700, 2100, 4650, 2000,
    3400, 2650, 2300, 4100, 4800, 4450, 2000, 2050,
], dtype=float)

DEFAULT_CPAS = np.array([
    100, 70, 90, 110, 60, 130, 120, 80,
    70, 130, 100, 110, 120, 90, 60, 80,
    130, 80, 110, 100, 90, 120, 60, 70,
    120, 60, 90, 70, 100, 110, 130, 80,
    120, 90, 70, 80, 100, 110, 60, 130,
    90, 100, 110, 80, 60, 70, 130, 120,
], dtype=float)

DEFAULT_CATEGORIES = np.arange(NUM_AGENTS) // NUM_AGENT_CATEGORY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate policies in one shared AuctionNet loop.")
    parser.add_argument("--config", type=str, default="configs/gcp-run-9-selective.yaml")
    parser.add_argument("--run-name", type=str, default=None,
                        help="PPO run name. If set, evaluates saved_models/ppo_<run-name>_best.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional PPO checkpoint base path or .zip path.")
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--fixed-alphas", type=float, nargs="*", default=[])
    parser.add_argument(
        "--auctionnet-baselines",
        nargs="*",
        choices=["iql", "td3_bc"],
        default=None,
        help="Pretrained AuctionNet scalar-alpha baselines to evaluate. "
             "Defaults to iql unless --skip-iql is set.",
    )
    parser.add_argument("--skip-ppo", action="store_true")
    parser.add_argument("--skip-iql", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_config(config_path: str) -> dict:
    with open(resolve_path(config_path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def configure_paths(auctionnet_path: str) -> Path:
    auctionnet_root = resolve_path(auctionnet_path)
    if not auctionnet_root.exists():
        raise FileNotFoundError(f"AuctionNet path not found: {auctionnet_root}")

    for path in reversed([PROJECT_ROOT, auctionnet_root, auctionnet_root / "strategy_train_env"]):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return auctionnet_root


def stub_optional_model_pv_generator() -> None:
    module_name = "simul_bidding_env.PvGenerator.ModelPvGen"
    if module_name in sys.modules:
        return

    module = types.ModuleType(module_name)

    class ModelPvGenerator:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Common eval uses NeurIPSPvGen, not ModelPvGenerator.")

    module.ModelPvGenerator = ModelPvGenerator
    sys.modules[module_name] = module


def get_winner(slot_pit: np.ndarray) -> np.ndarray:
    slot_pit_t = slot_pit.T
    num_pv = slot_pit_t.shape[0]
    winner = np.full((num_pv, 3), -1, dtype=int)
    for pos in range(1, 4):
        winning = np.argwhere(slot_pit_t == pos)
        if winning.size > 0:
            pv_idx, agent_idx = winning.T
            winner[pv_idx, pos - 1] = agent_idx
    return winner


def adjust_over_cost(bids: np.ndarray, over_cost_ratio: np.ndarray,
                     slot_coefficients: np.ndarray, winner_pit: np.ndarray) -> None:
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


@dataclass
class EvalState:
    tick: int = 0
    remaining_budget: float = 0.0
    total_spend: float = 0.0
    last_lwc: float = 0.0
    recent_xi: list[tuple[int, int]] = field(default_factory=list)


class PolicyAdapter:
    name: str

    def reset(self) -> None:
        pass

    def bid(self, tick: int, pv_values: np.ndarray, pvalue_sigmas: np.ndarray,
            history: dict[str, list], eval_state: EvalState) -> tuple[np.ndarray, dict[str, float]]:
        raise NotImplementedError


class FixedAlphaPolicy(PolicyAdapter):
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.name = f"fixed_alpha_{self.alpha:g}"

    def bid(self, tick: int, pv_values: np.ndarray, pvalue_sigmas: np.ndarray,
            history: dict[str, list], eval_state: EvalState) -> tuple[np.ndarray, dict[str, float]]:
        return self.alpha * pv_values, {"alpha": self.alpha, "keep_fraction": 1.0}


class AuctionNetStrategyPolicy(PolicyAdapter):
    def __init__(self, baseline: str, budget: float, cpa: float, category: int):
        if baseline == "iql":
            from simul_bidding_env.strategy.iql_bidding_strategy import IqlBiddingStrategy

            strategy_cls = IqlBiddingStrategy
            self.name = "auctionnet_pretrained_iql"
        elif baseline == "td3_bc":
            from simul_bidding_env.strategy.td3_bc_bidding_strategy import TD3_BCBiddingStrategy

            strategy_cls = TD3_BCBiddingStrategy
            self.name = "auctionnet_pretrained_td3_bc"
        else:
            raise ValueError(f"Unsupported AuctionNet baseline: {baseline}")

        self.strategy = strategy_cls(budget=budget, cpa=cpa, category=category)

    def reset(self) -> None:
        self.strategy.reset()

    def bid(self, tick: int, pv_values: np.ndarray, pvalue_sigmas: np.ndarray,
            history: dict[str, list], eval_state: EvalState) -> tuple[np.ndarray, dict[str, float]]:
        self.strategy.remaining_budget = eval_state.remaining_budget
        bids = self.strategy.bidding(
            tick,
            pv_values,
            pvalue_sigmas,
            history["pvalue_info"],
            history["bids"],
            history["auction"],
            history["impression"],
            history["least_winning_cost"],
        )
        bids = np.asarray(bids, dtype=np.float64).reshape(-1)
        alpha = float(np.sum(bids) / max(np.sum(pv_values), 1e-12))
        return bids, {"alpha": alpha, "keep_fraction": 1.0}


class IQLPolicy(AuctionNetStrategyPolicy):
    def __init__(self, budget: float, cpa: float, category: int):
        super().__init__("iql", budget, cpa, category)


class TD3BCPolicy(AuctionNetStrategyPolicy):
    def __init__(self, budget: float, cpa: float, category: int):
        super().__init__("td3_bc", budget, cpa, category)


class PPOPolicy(PolicyAdapter):
    def __init__(self, env_config: dict, agent_config: dict, run_name: str | None,
                 checkpoint: str | None):
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        from agents.ppo_agent import PPOAgent
        from environment.gym_wrapper import AuctionNetGymEnv

        if checkpoint is None:
            if run_name is None:
                raise ValueError("PPO evaluation requires --run-name or --checkpoint")
            model_path = PROJECT_ROOT / "saved_models" / f"ppo_{run_name}_best" / "best_model"
        else:
            model_path = resolve_path(checkpoint)
            if model_path.suffix == ".zip":
                model_path = model_path.with_suffix("")

        if not model_path.with_suffix(".zip").exists():
            raise FileNotFoundError(f"PPO checkpoint not found: {model_path}.zip")

        vec_env = DummyVecEnv([lambda: Monitor(AuctionNetGymEnv(env_config))])
        vecnorm_path = model_path.with_name("best_model_vecnormalize.pkl")
        if not vecnorm_path.exists():
            fallback = Path(str(model_path) + "_vecnormalize.pkl")
            vecnorm_path = fallback if fallback.exists() else vecnorm_path
        if vecnorm_path.exists():
            vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            print(f"Loaded PPO VecNormalize stats: {vecnorm_path}")
        else:
            print(f"WARNING: no PPO VecNormalize stats found near {model_path}")

        self.agent = PPOAgent(
            observation_space=vec_env.observation_space,
            action_space=vec_env.action_space,
            config=agent_config,
            env=vec_env,
        )
        self.agent.load(str(model_path), env=vec_env)
        self.env = vec_env
        self.max_bid_multiplier = float(env_config["max_bid_multiplier"])
        self.action_mode = str(env_config.get("action_mode", "scalar"))
        self.min_keep_fraction = float(env_config.get("min_keep_fraction", 0.05))
        self.num_ticks = int(env_config["num_ticks"])
        self.initial_budget = float(env_config["budget"])
        self.name = f"ppo_{run_name or model_path.name}"

    def _obs(self, tick: int, player_pvalues: np.ndarray, eval_state: EvalState) -> np.ndarray:
        budget_frac = eval_state.remaining_budget / self.initial_budget
        time_frac = tick / self.num_ticks
        mean_pval = float(player_pvalues.mean())
        std_pval = float(player_pvalues.std())
        norm_price = eval_state.last_lwc / self.initial_budget
        if eval_state.recent_xi:
            total_won = sum(w for w, _ in eval_state.recent_xi)
            total_pv = sum(t for _, t in eval_state.recent_xi)
            win_rate = total_won / total_pv if total_pv > 0 else 0.0
        else:
            win_rate = 0.0
        spend_frac = eval_state.total_spend / self.initial_budget
        pacing_ratio = spend_frac / time_frac if time_frac > 0 else 1.0
        return np.array(
            [budget_frac, time_frac, mean_pval, std_pval, norm_price, win_rate, pacing_ratio],
            dtype=np.float32,
        )

    def _decode_action(self, action: np.ndarray) -> tuple[float, float]:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        alpha_signal = float(np.clip(action_arr[0], -1.0, 1.0))
        alpha = ((alpha_signal + 1.0) / 2.0) * self.max_bid_multiplier
        keep_fraction = 1.0
        if self.action_mode == "selective_topk":
            keep_signal = float(np.clip(action_arr[1], -1.0, 1.0)) if action_arr.size > 1 else 1.0
            keep_raw = (keep_signal + 1.0) / 2.0
            keep_fraction = self.min_keep_fraction + keep_raw * (1.0 - self.min_keep_fraction)
            keep_fraction = float(np.clip(keep_fraction, self.min_keep_fraction, 1.0))
        return float(alpha), keep_fraction

    def bid(self, tick: int, pv_values: np.ndarray, pvalue_sigmas: np.ndarray,
            history: dict[str, list], eval_state: EvalState) -> tuple[np.ndarray, dict[str, float]]:
        obs = self._obs(tick, pv_values, eval_state)
        policy_obs = self.env.normalize_obs(obs)
        action, _, _ = self.agent.select_action(policy_obs, deterministic=True)
        alpha, keep_fraction = self._decode_action(action)
        bids = alpha * pv_values
        if self.action_mode == "selective_topk" and keep_fraction < 1.0:
            cutoff = float(np.quantile(pv_values, 1.0 - keep_fraction))
            bids = np.where(pv_values >= cutoff, bids, 0.0)
        return bids, {"alpha": alpha, "keep_fraction": keep_fraction}


def build_competitors(player_index: int):
    from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy

    agents = []
    for i in range(NUM_AGENTS):
        if i == player_index:
            agents.append(None)
            continue
        agent = PidBiddingStrategy(exp_tempral_ratio=np.ones(48))
        agent.budget = float(DEFAULT_BUDGETS[i])
        agent.cpa = float(DEFAULT_CPAS[i])
        agent.category = int(DEFAULT_CATEGORIES[i])
        agent.name = f"agent_{i}"
        agents.append(agent)
    return agents


def shaped_reward(env_config: dict, conversion_value: float, cost: float,
                  total_spend: float, tick: int) -> float:
    reward_value_scale = float(env_config.get("reward_value_scale", 1.0))
    reward_lambda_cost = float(env_config.get("reward_lambda_cost", 1.0))
    reward_beta_pacing = float(env_config.get("reward_beta_pacing", 0.0))
    budget = float(env_config["budget"])
    num_ticks = int(env_config["num_ticks"])
    time_frac = (tick + 1) / num_ticks
    pacing_ratio = (total_spend / budget) / time_frac
    return (
        reward_value_scale * conversion_value
        - reward_lambda_cost * cost
        + reward_beta_pacing * min(pacing_ratio, 1.0)
    )


def evaluate_policy(policy: PolicyAdapter, env_config: dict, episodes: int) -> dict[str, Any]:
    from simul_bidding_env.Environment.BiddingEnv import BiddingEnv
    from simul_bidding_env.PvGenerator.NeurIPSPvGen import NeurIPSPvGen

    player_index = int(env_config["player_index"])
    player_budget = float(env_config["budget"])
    num_ticks = int(env_config["num_ticks"])
    num_episode = int(env_config.get("num_episode", episodes))
    pv_num = int(env_config["pv_num"])
    min_remaining_budget = float(env_config["min_remaining_budget"])

    env = BiddingEnv(min_remaining_budget=min_remaining_budget)
    competitors = build_competitors(player_index)

    totals = {
        "conversions": 0.0,
        "cost": 0.0,
        "slot_wins": 0,
        "exposures": 0,
        "auctions": 0,
        "shaped_rewards": [],
        "alphas": [],
        "keep_fractions": [],
    }

    for ep in range(episodes):
        episode_id = ep % num_episode
        env.reset(episode=episode_id)
        pv_gen = NeurIPSPvGen(
            episode=episode_id,
            num_tick=num_ticks,
            num_agent=NUM_AGENTS,
            num_agent_category=NUM_AGENT_CATEGORY,
            num_category=NUM_CATEGORY,
            pv_num=pv_num,
        )

        policy.reset()
        for i, agent in enumerate(competitors):
            if agent is None:
                continue
            agent.reset()
            agent.remaining_budget = float(DEFAULT_BUDGETS[i])

        eval_state = EvalState(remaining_budget=player_budget)
        history_all = {
            "pvalue_info": [],
            "bids": [],
            "auction": [],
            "impression": [],
            "least_winning_cost": [],
        }
        ep_reward = 0.0

        for tick in range(num_ticks):
            pv_values = pv_gen.pv_values[tick]
            pvalue_sigmas = pv_gen.pValueSigmas[tick]
            remaining_budgets = np.array([
                eval_state.remaining_budget if i == player_index else (
                    agent.remaining_budget if agent is not None else 0.0
                )
                for i, agent in enumerate(competitors)
            ])

            all_bids = []
            for i, agent in enumerate(competitors):
                if i == player_index:
                    player_history = {
                        "pvalue_info": [x[i] for x in history_all["pvalue_info"]],
                        "bids": [x[i] for x in history_all["bids"]],
                        "auction": [x[i] for x in history_all["auction"]],
                        "impression": [x[i] for x in history_all["impression"]],
                        "least_winning_cost": history_all["least_winning_cost"],
                    }
                    bids_i, action_info = policy.bid(
                        tick,
                        pv_values[:, i],
                        pvalue_sigmas[:, i],
                        player_history,
                        eval_state,
                    )
                    totals["alphas"].append(float(action_info["alpha"]))
                    totals["keep_fractions"].append(float(action_info["keep_fraction"]))
                elif agent is None or remaining_budgets[i] < min_remaining_budget:
                    bids_i = np.zeros(pv_values.shape[0])
                else:
                    bids_i = agent.bidding(
                        tick,
                        pv_values[:, i],
                        pvalue_sigmas[:, i],
                        [x[i] for x in history_all["pvalue_info"]],
                        [x[i] for x in history_all["bids"]],
                        [x[i] for x in history_all["auction"]],
                        [x[i] for x in history_all["impression"]],
                        history_all["least_winning_cost"],
                    )
                all_bids.append(np.asarray(bids_i, dtype=np.float64))

            bids = np.array(all_bids).T
            bids = np.clip(bids, 0, None)

            winner_pit = None
            while True:
                xi_pit, slot_pit, cost_pit, is_exposed_pit, conversion_action_pit, \
                    lwc_pit, market_price_pit = env.simulate_ad_bidding(
                        pv_values, pvalue_sigmas, bids
                    )
                real_cost = (cost_pit * is_exposed_pit).sum(axis=1)
                over_cost_ratio = np.maximum(
                    (real_cost - remaining_budgets) / (real_cost + 1e-4), 0
                )
                winner_pit = get_winner(slot_pit)
                if over_cost_ratio.max() == 0:
                    break
                adjust_over_cost(bids, over_cost_ratio, env.slot_coefficients, winner_pit)

            player_cost = float(real_cost[player_index])
            player_conversions = float(conversion_action_pit[player_index].sum())
            player_slot_wins = int(xi_pit[player_index].sum())
            player_exposures = int(is_exposed_pit[player_index].sum())
            player_auctions = int(xi_pit.shape[1])

            eval_state.remaining_budget -= player_cost
            eval_state.total_spend += player_cost
            eval_state.last_lwc = float(lwc_pit.mean())
            eval_state.tick = tick + 1
            eval_state.recent_xi.append((player_slot_wins, player_auctions))
            if len(eval_state.recent_xi) > 3:
                eval_state.recent_xi.pop(0)

            for i, agent in enumerate(competitors):
                if agent is not None and i != player_index:
                    agent.remaining_budget -= real_cost[i]

            totals["conversions"] += player_conversions
            totals["cost"] += player_cost
            totals["slot_wins"] += player_slot_wins
            totals["exposures"] += player_exposures
            totals["auctions"] += player_auctions
            ep_reward += shaped_reward(
                env_config, player_conversions, player_cost, eval_state.total_spend, tick
            )

            pvalue_info = np.stack((pv_values.T, pvalue_sigmas.T), axis=-1)
            history_all["pvalue_info"].append(pvalue_info)
            history_all["bids"].append(bids.T)
            auction_info = np.stack((xi_pit, slot_pit, cost_pit), axis=-1)
            history_all["auction"].append(auction_info)
            impression_info = np.stack((is_exposed_pit, conversion_action_pit), axis=-1)
            history_all["impression"].append(impression_info)
            history_all["least_winning_cost"].append(lwc_pit)

            if eval_state.remaining_budget < min_remaining_budget:
                break

        totals["shaped_rewards"].append(ep_reward)

    conversions = totals["conversions"]
    cost = totals["cost"]
    auctions = totals["auctions"]
    shaped_rewards = np.array(totals["shaped_rewards"], dtype=float)
    alphas = np.array(totals["alphas"], dtype=float)
    keep_fractions = np.array(totals["keep_fractions"], dtype=float)

    return {
        "policy": policy.name,
        "episodes": episodes,
        "conversions": conversions,
        "cost": cost,
        "budget_utilization": cost / (episodes * player_budget),
        "aggregate_cpa": cost / conversions if conversions > 0 else None,
        "conversion_roi": conversions / cost if cost > 0 else 0.0,
        "slot_win_rate": totals["slot_wins"] / auctions if auctions > 0 else 0.0,
        "exposure_rate": totals["exposures"] / auctions if auctions > 0 else 0.0,
        "shaped_ep_reward_mean": float(shaped_rewards.mean()) if shaped_rewards.size else 0.0,
        "shaped_ep_reward_std": float(shaped_rewards.std(ddof=1)) if shaped_rewards.size > 1 else 0.0,
        "alpha_mean": float(alphas.mean()) if alphas.size else 0.0,
        "alpha_std": float(alphas.std(ddof=1)) if alphas.size > 1 else 0.0,
        "keep_fraction_mean": float(keep_fractions.mean()) if keep_fractions.size else 1.0,
        "keep_fraction_std": float(keep_fractions.std(ddof=1)) if keep_fractions.size > 1 else 0.0,
    }


def print_result(result: dict[str, Any]) -> None:
    print(f"\n=== {result['policy']} ===")
    print(f"episodes:            {result['episodes']}")
    print(f"conversions:         {result['conversions']:.6f}")
    print(f"cost:                {result['cost']:.6f}")
    print(f"budget utilization:  {result['budget_utilization']:.2%}")
    cpa = result["aggregate_cpa"]
    print(f"aggregate CPA:       {cpa:.6f}" if cpa is not None else "aggregate CPA:       n/a")
    print(f"conversion ROI:      {result['conversion_roi']:.6f}")
    print(f"slot win rate:       {result['slot_win_rate']:.2%}")
    print(f"exposure rate:       {result['exposure_rate']:.2%}")
    print(
        "shaped ep reward:    "
        f"{result['shaped_ep_reward_mean']:.2f} +/- {result['shaped_ep_reward_std']:.2f}"
    )
    print(f"alpha mean/std:      {result['alpha_mean']:.2f} / {result['alpha_std']:.2f}")
    print(
        "keep frac mean/std:  "
        f"{result['keep_fraction_mean']:.3f} / {result['keep_fraction_std']:.3f}"
    )


def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    env_config = config["environment"]
    agent_config = config.get("agent", {})

    auctionnet_root = configure_paths(env_config["auctionnet_path"])
    os.chdir(auctionnet_root)
    stub_optional_model_pv_generator()

    policies: list[PolicyAdapter] = []
    if not args.skip_ppo:
        policies.append(PPOPolicy(env_config, agent_config, args.run_name, args.checkpoint))

    auctionnet_baselines = args.auctionnet_baselines
    if auctionnet_baselines is None:
        auctionnet_baselines = [] if args.skip_iql else ["iql"]
    elif args.skip_iql:
        auctionnet_baselines = [b for b in auctionnet_baselines if b != "iql"]

    for baseline in auctionnet_baselines:
        policies.append(AuctionNetStrategyPolicy(
            baseline,
            budget=float(env_config["budget"]),
            cpa=float(DEFAULT_CPAS[int(env_config["player_index"])]),
            category=int(DEFAULT_CATEGORIES[int(env_config["player_index"])]),
        ))
    for alpha in args.fixed_alphas:
        policies.append(FixedAlphaPolicy(alpha))

    results = []
    for policy in policies:
        result = evaluate_policy(policy, env_config, args.episodes)
        results.append(result)
        print_result(result)

    if args.output_json:
        output_path = resolve_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=json_default)
        print(f"\nWrote metrics JSON: {output_path}")


if __name__ == "__main__":
    main()
