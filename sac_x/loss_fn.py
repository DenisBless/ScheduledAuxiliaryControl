import torch
import torch.nn.functional as F
from torch.multiprocessing import current_process


class Retrace:
    def __init__(self, num_actions, num_intentions):
        super(Retrace, self).__init__()
        self.num_actions = num_actions
        self.num_intentions = num_intentions

    def __call__(self,
                 Q,
                 expected_target_Q,
                 target_Q,
                 rewards,
                 target_policy_probs,
                 behaviour_policy_probs,
                 gamma=0.99,
                 logger=None):
        """
        Implementation of Retrace loss ((http://arxiv.org/abs/1606.02647)) in PyTorch.

        Args:
            Q: State-Action values.
            Torch tensor with shape `[#intentions, T]`

            expected_target_Q: 𝔼_π Q(s_t,.) (from the fixed critic network)
            Torch tensor with shape `[#intentions, T]`

            target_Q: State-Action values from target network.
            Torch tensor with shape `[#intentions, T]`

            rewards: Holds rewards for taking an action in the environment.
            Torch tensor with shape `[#intentions, T]`

            target_policy_probs: Probability of target policy π(a|s)
            Torch tensor with shape `[#intentions, T]`

            behaviour_policy_probs: Probability of behaviour policy b(a|s)
            Torch tensor with shape `[1, T]`

            gamma: Discount factor

        Returns:

            Computes the retrace loss recursively according to
            L = 𝔼_τ[(Q_t - Q_ret_t)^2]
            Q_ret_t = r_t + γ * (𝔼_π_target [Q(s_t+1,•)] + c_t+1 * Q_π_target(s_t+1,a_t+1)) + γ * c_t+1 * Q_ret_t+1

            with trajectory τ = {(s_0, a_0, r_0),..,(s_k, a_k, r_k)}
        """

        # adjust and check dimensions
        Q.squeeze_(dim=-1)
        target_Q.squeeze_(dim=-1)
        expected_target_Q.squeeze_(dim=-1)
        T = Q.shape[1]
        assert Q.shape == target_Q.shape == expected_target_Q.shape == rewards.shape == target_policy_probs.shape == \
               [self.num_intentions, T]
        assert behaviour_policy_probs.shape == [1, T]

        with torch.no_grad():
            # We don't want gradients from computing Q_ret, since:
            # ∇φ (Q - Q_ret)^2 ∝ (Q - Q_ret) * ∇φ Q

            c_ret = self.calc_retrace_weights(target_policy_probs, behaviour_policy_probs)
            Q_ret = torch.zeros_like(Q, dtype=torch.float)  # (#intentions,T)

            Q_ret[:, -1] = target_Q[:, -1]
            for t in reversed(range(1, T)):
                Q_ret[:, t - 1] = rewards[:, t - 1] + gamma * c_ret[:, t] * (Q_ret[:, t] - target_Q[:, t]) + \
                                  gamma * expected_target_Q[:, t]

            if logger is not None:
                logger.add_histogram(tag="retrace/ratio", values=c_ret)
                logger.add_histogram(tag="retrace/behaviour", values=behaviour_policy_probs)
                logger.add_histogram(tag="retrace/target", values=target_policy_probs)

                logger.add_scalar(tag="retace/Qret-targetQ mean", scalar_value=(Q_ret - target_Q).mean())
                logger.add_scalar(tag="retace/Qret-targetQ std", scalar_value=(Q_ret - target_Q).std())
                logger.add_scalar(tag="retrace/E[targetQ] mean", scalar_value=expected_target_Q.mean())
                logger.add_scalar(tag="retrace/E[targetQ] std", scalar_value=expected_target_Q.std())

        return F.mse_loss(Q, Q_ret)

    def calc_retrace_weights(self, target_policy_logprob, behaviour_policy_logprob):
        """
        Calculates the retrace weights (truncated importance weights) c according to:
        c_t = min(1, π_target(a_t|s_t) / b(a_t|s_t)) where:
        π_target: target policy probabilities
        b: behaviour policy probabilities

        Args:
            target_policy_logprob: log π_target(a_t|s_t)
            behaviour_policy_logprob: log b(a_t|s_t)

        Returns:
            retrace weights c
        """
        assert target_policy_logprob.shape == behaviour_policy_logprob.shape, \
            "Error, shape mismatch. Shapes: target_policy_logprob: " \
            + str(target_policy_logprob.shape) + " mean: " + str(behaviour_policy_logprob.shape)

        log_retrace_weights = (target_policy_logprob - behaviour_policy_logprob).clamp(max=0)
        retrace_weights = log_retrace_weights.exp()
        assert not torch.isnan(log_retrace_weights).any(), "Error, a least one NaN value found in retrace weights."
        # return log_retrace_weights.exp()
        return retrace_weights


class ActorLoss:
    def __init__(self, num_intentions, alpha=0):
        """
        Loss function for the actor.
        Args:
            alpha: entropy regularization parameter.
        """
        self.num_intentions = num_intentions
        self.alpha = alpha

    def __call__(self, Q, action_log_prob):
        """
        Computes the loss of the actor according to
        L = 𝔼_π [Q(a,s) - α log(π(a|s)]
        Args:
            Q: Q(a,s)
            action_log_prob: log(π(a|s)

        Returns:
            Scalar actor loss value
        """
        assert Q.dim() == action_log_prob.dim()
        return (self.alpha * action_log_prob - Q).mean(dim=-1).sum()
