"""
Evaluation metrics for real-time ad bidding agents.
"""

from typing import List, Tuple, Dict


Step = Tuple  # (obs, action, reward, info)


def _extract_info(step: Step) -> Dict:
    """Helper: get info dict (handle vec_env case)."""
    info = step[3]
    if isinstance(info, list):  # DummyVecEnv
        info = info[0]
    return info


"""
All metrics are computed from a trajectory — a sequence of (observation,
action, reward, info) tuples collected by rolling out an agent in the
AuctionNetGymEnv. The info dict from each step contains the raw auction
outcome data (clearing price, conversion value, win/loss indicator) needed
to compute these metrics independently of the reward shaping used during
training.

Core metrics:
    - ROI: measures capital efficiency (are we winning valuable auctions?)
    - Budget Utilization: measures spend pacing (are we using our budget?)
    - Win Rate: diagnostic metric for how often we outbid competitors
"""





# =========================
# Core metrics
# =========================

def compute_total_value(trajectory: List[Step]) -> float:
    return sum(_extract_info(step)["conversion_value"] for step in trajectory)



def compute_total_cost(trajectory: List[Step]) -> float:
    return sum(_extract_info(step)["cost"] for step in trajectory)



def compute_profit(trajectory: List[Step]) -> float:
    return compute_total_value(trajectory) - compute_total_cost(trajectory)


def compute_roi(trajectory: List[Step]) -> float:
    """Compute Return on Investment (ROI) over a full episode trajectory.

    ROI measures the capital efficiency of the bidding strategy:
        ROI = total_conversion_value_won / total_budget_spent

    A value > 1.0 means the agent generated more value than it spent.
    Higher is better. Agents that never win auctions will have ROI = 0.
    Agents that win low-value auctions at high prices will have ROI < 1.

    Args:
        trajectory (list): List of step tuples (obs, action, reward, info)
            from one full episode. Each info dict must contain:
                - 'conversion_value' (float): value of the won impression, 0 if lost
                - 'cost' (float): amount spent (clearing price if won, else 0)

    Returns:
        roi (float): Total conversion value divided by total spend.
            Returns 0.0 if total spend is zero (no auctions won).
    """
    total_value = compute_total_value(trajectory)
    total_cost = compute_total_cost(trajectory)

    if total_cost == 0:
        return 0.0
    return total_value / total_cost


def compute_budget_utilization(trajectory: List[Step]) -> float:
    """Compute budget utilization over a full episode trajectory.

    Budget utilization measures how effectively the agent paces its spend:
        utilization = total_spend / total_budget_available

    The target is ~1.0 (fully utilized). Under-spending means missed
    opportunities; over-spending is disallowed by the environment.
    Very low utilization often indicates the agent's bids are too low
    to be competitive.

    Args:
        trajectory (list): List of step tuples (obs, action, reward, info)
            from one full episode. Each info dict must contain:
                - 'cost' (float): amount spent at this step
            The first info dict (or episode metadata) must contain:
                - 'total_budget' (float): the episode's starting budget

    Returns:
        utilization (float): Fraction of budget spent, in [0, 1].
    """
    total_cost = compute_total_cost(trajectory)

    first_info = _extract_info(trajectory[0])
    last_info = _extract_info(trajectory[-1])

    remaining_budget_end = last_info["remaining_budget"]

    # Infer initial budget
    initial_budget = remaining_budget_end + total_cost

    if initial_budget <= 0:
        return 0.0

    return total_cost / initial_budget

def compute_win_rate(trajectory: List[Step]) -> float:
    """Compute auction win rate over a full episode trajectory.

    Win rate is the fraction of auctions entered where the agent submitted
    the highest bid and won the impression:
        win_rate = auctions_won / auctions_entered

    This is a diagnostic metric — a high win rate is not necessarily good
    (it may mean the agent is overbidding). Used alongside ROI and budget
    utilization to diagnose failure modes (e.g., low win rate + low budget
    utilization suggests bids are too low).

    Args:
        trajectory (list): List of step tuples (obs, action, reward, info)
            from one full episode. Each info dict must contain:
                - 'won' (bool): True if the agent won this auction step.

    Returns:
        win_rate (float): Fraction of auctions won, in [0, 1].
    """
   
    wins = sum(float(_extract_info(step).get("auctions_won", 0.0)) for step in trajectory)
    participated = sum(float(_extract_info(step).get("auctions_participated", 0.0)) for step in trajectory)
    if participated <= 0:
        return 0.0
    return wins / participated


def compute_avg_cost(trajectory: List[Step]) -> float:
    costs = [_extract_info(step)["cost"] for step in trajectory]
    return sum(costs) / len(costs) if costs else 0.0


def compute_avg_value(trajectory: List[Step]) -> float:
    values = [_extract_info(step)["conversion_value"] for step in trajectory]
    return sum(values) / len(values) if values else 0.0



def compute_all_metrics(trajectory: List[Step]) -> dict:
    return {
        "total_value": compute_total_value(trajectory),
        "total_cost": compute_total_cost(trajectory),
        "profit": compute_profit(trajectory),
        "roi": compute_roi(trajectory),
        "budget_utilization": compute_budget_utilization(trajectory),
        "win_rate": compute_win_rate(trajectory),
        "avg_cost": compute_avg_cost(trajectory),
        "avg_value": compute_avg_value(trajectory),
    }