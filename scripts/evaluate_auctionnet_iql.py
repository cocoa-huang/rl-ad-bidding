"""
Evaluate AuctionNet's pretrained IQL strategy under this project's config.

This is a system-level baseline for comparing our PPO policy against the
official AuctionNet offline RL agent. It intentionally uses AuctionNet's
pretrained IQL model and 16-dimensional engineered state, while matching the
important environment knobs from the selected PPO config: player index, budget,
PV count, and number of ticks.

Usage:
    python scripts/evaluate_auctionnet_iql.py --config configs/gcp-run-9-selective.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import gin
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained AuctionNet IQL with this project's env config."
    )
    parser.add_argument("--config", type=str, default="configs/gcp-run-9-selective.yaml")
    parser.add_argument("--episodes", type=int, default=None)
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

    paths = [
        PROJECT_ROOT,
        auctionnet_root,
        auctionnet_root / "strategy_train_env",
    ]
    for path in reversed(paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return auctionnet_root


def require_iql_assets(auctionnet_root: Path) -> None:
    iql_dir = auctionnet_root / "simul_bidding_env" / "strategy" / "official_agent" / "IQLtest"
    required = [iql_dir / "iql_model.pth", iql_dir / "normalize_dict.pkl"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing pretrained AuctionNet IQL asset(s):\n" + "\n".join(missing)
        )


def patch_auctionnet_for_project_env(env_config: dict) -> None:
    """Patch AuctionNet's controller to match this project's comparison setup."""
    from simul_bidding_env.Controller.Controller import Controller
    from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy

    player_index = int(env_config["player_index"])
    player_budget = float(env_config["budget"])
    num_ticks = int(env_config["num_ticks"])

    def initialize_pid_agents(self):
        return [
            PidBiddingStrategy(exp_tempral_ratio=np.ones(num_ticks))
            for _ in range(self.num_agent)
        ]

    def calculate_budget_with_project_player(self):
        budgets = np.array([
            2900, 4350, 3000, 2400, 4800, 2000, 2050, 3500,
            4600, 2000, 2800, 2350, 2050, 2900, 4750, 3450,
            2000, 3500, 2200, 2700, 3100, 2100, 4850, 4100,
            2000, 4800, 3050, 4250, 2850, 2250, 2000, 3900,
            2000, 3250, 4450, 3550, 2700, 2100, 4650, 2000,
            3400, 2650, 2300, 4100, 4800, 4450, 2000, 2050,
        ], dtype=float)
        budgets[player_index] = player_budget
        return budgets.tolist()

    Controller.initialize_agents = initialize_pid_agents
    Controller.calculate_budget = calculate_budget_with_project_player


def run_iql_eval(env_config: dict, episodes: int) -> dict:
    from simul_bidding_env.strategy.iql_bidding_strategy import IqlBiddingStrategy
    import run.run_test as auctionnet_run_test

    patch_auctionnet_for_project_env(env_config)

    player_budget = float(env_config["budget"])
    player_index = int(env_config["player_index"])
    num_ticks = int(env_config["num_ticks"])
    pv_num = int(env_config["pv_num"])

    # The official IQL strategy is a scalar alpha policy trained by AuctionNet.
    # Player 0 has CPA 100/category 0 in this project's PPO experiments.
    auctionnet_run_test.initialize_player_agent = lambda: IqlBiddingStrategy(
        budget=player_budget,
        cpa=100,
        category=0,
    )

    gin.clear_config()
    gin.bind_parameter("Controller.num_agent_category", 8)
    gin.bind_parameter("Controller.num_category", 6)
    gin.bind_parameter("Controller.num_tick", num_ticks)
    gin.bind_parameter("Controller.pv_num", pv_num)

    result = auctionnet_run_test.run_test(
        generate_log=False,
        num_episode=episodes,
        num_tick=num_ticks,
        player_index=player_index,
    )

    total_value = float(result["reward"])
    total_cost = float(result["allCost"])
    budget_utilization = float(result["budget_consumer_ratio"])
    win_rate = float(result["win_pv_ratio"])

    return {
        "policy": "auctionnet_pretrained_iql",
        "episodes": episodes,
        "player_index": player_index,
        "budget": player_budget,
        "pv_num": pv_num,
        "num_ticks": num_ticks,
        "total_value": total_value,
        "total_cost": total_cost,
        "roi": total_value / total_cost if total_cost > 0 else 0.0,
        "budget_utilization": budget_utilization,
        "win_rate": win_rate,
        "cpa": float(result["cpa"]),
        "cpa_constraint": float(result["cpaConstraint"]),
        "score": float(result["score"]),
        "raw_result": result,
    }


def print_metrics(metrics: dict) -> None:
    print("\n=== AuctionNet Pretrained IQL Baseline ===")
    print(f"episodes:            {metrics['episodes']}")
    print(f"player_index:        {metrics['player_index']}")
    print(f"budget:              {metrics['budget']:.2f}")
    print(f"pv_num:              {metrics['pv_num']}")
    print(f"total value:         {metrics['total_value']:.6f}")
    print(f"total cost:          {metrics['total_cost']:.6f}")
    print(f"ROI value/cost:      {metrics['roi']:.6f}")
    print(f"budget utilization:  {metrics['budget_utilization']:.2%}")
    print(f"win/exposure rate:   {metrics['win_rate']:.2%}")
    print(f"CPA:                 {metrics['cpa']:.6f}")
    print(f"CPA constraint:      {metrics['cpa_constraint']:.6f}")
    print(f"score:               {metrics['score']:.6f}")
    print("\nNote: AuctionNet IQL is a scalar-alpha policy, so it does not use")
    print("gcp-run-9's selective_topk second action dimension.")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    env_config = config["environment"]

    auctionnet_root = configure_paths(env_config["auctionnet_path"])
    require_iql_assets(auctionnet_root)

    # AuctionNet imports and relative model paths assume its repo root as cwd.
    os.chdir(auctionnet_root)

    episodes = args.episodes
    if episodes is None:
        episodes = int(env_config.get("num_episode", 48))

    metrics = run_iql_eval(env_config, episodes)
    print_metrics(metrics)

    if args.output_json:
        output_path = resolve_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nWrote metrics JSON: {output_path}")


if __name__ == "__main__":
    main()
