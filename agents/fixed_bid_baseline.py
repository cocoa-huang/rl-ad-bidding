"""
Fixed-bid baseline agent for real-time ad bidding.

The simplest possible bidding strategy: always submit the same fixed bid
regardless of context, budget remaining, or market conditions. Serves as a
lower-bound baseline that any learned policy should comfortably beat.

Used to sanity-check that the environment and evaluation pipeline are working
correctly before running more complex RL experiments.
"""


class FixedBidBaseline:
    """Baseline agent that always bids a constant fixed amount.

    No learning occurs. The bid amount is set at construction time and
    remains constant throughout the episode. Useful as a sanity-check
    lower bound: a learned agent should always outperform this on ROI.

    Attributes:
        fixed_bid (float): The constant bid value submitted at every step.
    """

    def __init__(self, fixed_bid: float):
        """Initialize the baseline with a constant bid amount.

        Args:
            fixed_bid (float): The bid amount to submit at every auction step.
                Should be in the same units as the environment's action space.
        """
        raise NotImplementedError

    def select_action(self, observation):
        """Return the fixed bid, ignoring the observation entirely.

        Args:
            observation (np.ndarray): Current environment observation
                (ignored by this baseline).

        Returns:
            action (float): The constant fixed_bid value.
        """
        raise NotImplementedError

    def save(self, path: str):
        """Serialize the fixed bid value to disk.

        Args:
            path (str): File path (relative) to save the config.
        """
        raise NotImplementedError

    def load(self, path: str):
        """Load a saved fixed bid value from disk.

        Args:
            path (str): File path (relative) to the saved config.
        """
        raise NotImplementedError
