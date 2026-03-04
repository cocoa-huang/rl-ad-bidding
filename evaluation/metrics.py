"""
Evaluation metrics for real-time ad bidding agents.

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


def compute_roi(trajectory: list) -> float:
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
    raise NotImplementedError


def compute_budget_utilization(trajectory: list) -> float:
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
    raise NotImplementedError


def compute_win_rate(trajectory: list) -> float:
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
    raise NotImplementedError
