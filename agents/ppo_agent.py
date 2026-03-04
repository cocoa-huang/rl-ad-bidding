"""
Proximal Policy Optimization (PPO) agent for real-time ad bidding.

Implements the clipped-objective PPO algorithm (Schulman et al., 2017) adapted
for the continuous action space of the AuctionNetGymEnv. The actor outputs a
Gaussian distribution over bid multipliers; the critic estimates the value
function used for advantage estimation via GAE-lambda.

Reference:
    Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
    Proximal Policy Optimization Algorithms. arXiv:1707.06347.
"""


class PPOAgent:
    """PPO agent with a clipped surrogate objective.

    Maintains separate actor (policy) and critic (value function) networks.
    Updates are performed on mini-batches collected over a fixed rollout
    horizon, with clipping to constrain the policy update step size.

    Attributes:
        config (dict): Hyperparameters from configs/YAML (lr, gamma, batch_size,
            clip_epsilon, n_epochs, gae_lambda, etc.).
        actor (torch.nn.Module): Policy network mapping observations to action
            distribution parameters.
        critic (torch.nn.Module): Value network mapping observations to scalar
            state-value estimates.
        optimizer (torch.optim.Optimizer): Shared or separate optimizers for
            actor and critic.
    """

    def __init__(self, observation_space, action_space, config: dict):
        """Initialize actor and critic networks and optimizer.

        Args:
            observation_space (gymnasium.Space): Observation space from the env,
                used to determine input dimensionality.
            action_space (gymnasium.Space): Action space from the env, used to
                determine output dimensionality and action bounds.
            config (dict): Hyperparameter config from configs/default.yaml
                under the 'agent' section.
        """
        raise NotImplementedError

    def select_action(self, observation):
        """Sample an action from the current policy given an observation.

        Args:
            observation (np.ndarray): Current environment observation.

        Returns:
            action (np.ndarray): Sampled action (bid multiplier).
            log_prob (float): Log-probability of the sampled action under the
                current policy, needed for the PPO surrogate loss.
            value (float): Critic's value estimate for this observation.
        """
        raise NotImplementedError

    def update(self, rollout_buffer):
        """Perform one PPO update step on the collected rollout buffer.

        Computes GAE-lambda advantages, then iterates over the buffer for
        n_epochs mini-batch gradient updates using the clipped surrogate
        objective plus value loss and entropy bonus.

        Args:
            rollout_buffer: Object containing collected transitions
                (observations, actions, rewards, log_probs, values, dones).

        Returns:
            metrics (dict): Diagnostic scalars including policy_loss,
                value_loss, entropy, and approx_kl for W&B logging.
        """
        raise NotImplementedError

    def save(self, path: str):
        """Serialize agent weights to disk.

        Args:
            path (str): File path (relative) to save the checkpoint.
                Checkpoints are stored under saved_models/.
        """
        raise NotImplementedError

    def load(self, path: str):
        """Load agent weights from a checkpoint file.

        Args:
            path (str): File path (relative) to the checkpoint to load.
        """
        raise NotImplementedError
