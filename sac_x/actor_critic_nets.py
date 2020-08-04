import torch
import torch.nn as nn
import numpy as np

from typing import List, Tuple
from torch.distributions.normal import Normal


class Base(nn.Module):
    def __init__(self,
                 base_layer_dims: List,
                 non_linearity: nn.Module = nn.ReLU()):
        super(Base, self).__init__()

        base_modules = []
        for i in range(len(base_layer_dims) - 1):
            base_modules.append(torch.nn.Linear(base_layer_dims[i], base_layer_dims[i + 1]))
            base_modules.append(non_linearity)

        self.base_model = nn.Sequential(*base_modules)
        self.init_weights(self.base_model)

    def copy_params(self, source_network: nn.Module) -> None:
        """
        Copy the parameters from the source network to the current network.

        Args:
            source_network: Network to copy parameters from

        Returns:
            No return value
        """
        for param, source_param in zip(self.parameters(), source_network.parameters()):
            param.data.copy_(source_param.data)

    def freeze_net(self) -> None:
        """
        Deactivate gradient
            Computation for the network

        Returns:
            No return value
        """
        for params in self.parameters():
            params.requires_grad = False

    def is_shared(self) -> bool:
        """
        Checks if the network parameter are shared.

        Returns:
            True if shared
        """
        for params in self.parameters():
            if not params.is_shared():
                return False
        return True

    @staticmethod
    def init_weights(module: nn.Module) -> None:
        """
        Orthogonal initialization of the weights. Sets initial bias to zero.

        Args:
            module: Network to initialize weights.

        Returns:
            No return value

        """
        if type(module) == nn.Linear:
            nn.init.orthogonal_(module.weight)
            module.bias.data.fill_(0.0)


class Actor(Base):
    def __init__(self,
                 num_intentions: int,
                 num_actions: int,
                 num_obs: int,
                 base_layer_dims: List = None,
                 intention_layer_dims: List = None,
                 std_init: float = -2.,
                 non_linearity: nn.Module = nn.ReLU(),
                 eps: float = 1e-6,
                 logger=None):

        if base_layer_dims is None:
            base_layer_dims = [64, 64]
        if intention_layer_dims is None:
            intention_layer_dims = [32]
        assert std_init > 0

        super(Actor, self).__init__(base_layer_dims=[num_obs] + base_layer_dims,
                                    non_linearity=non_linearity)

        self.num_intentions = num_intentions
        self.num_actions = num_actions
        self.log_std_init = np.log(std_init)
        self.eps = eps
        self.logger = logger

        # Create a model for a intention net
        intention_modules = []
        for i in range(len(intention_layer_dims) - 1):
            intention_modules.append(nn.Linear(intention_layer_dims[i], intention_layer_dims[i + 1]))
            intention_modules.append(non_linearity)
        intention_modules.append(nn.Linear(intention_layer_dims[-1], num_actions))

        # Create all intention nets
        self.intention_nets = []
        for i in range(num_intentions):
            intention_net = nn.Sequential(*intention_modules[:-1])  # Remove last non-linearity
            self.init_weights(intention_net)
            self.intention_nets.append(intention_net)

        # state independent action noise
        self.log_std = torch.nn.Parameter(torch.ones(num_intentions, num_actions) * self.log_std_init)

    def __call__(self, x, intention_idx):
        return self.predict(x, intention_idx)

    def predict(self, x, intention_idx=None):
        assert 0 <= intention_idx < self.num_intentions
        assert list(self.log_std[intention_idx].shape) == [1, self.num_actions]

        x = self.base_model(x)

        if intention_idx is None:
            means = torch.tensor([self.num_intentions, self.num_actions], dtype=torch.float32)
            for i in range(self.num_intentions):
                means[i, :] = self.intention_nets[i](x)
            return means, self.log_std

        else:
            mean = self.intention_nets[intention_idx](x)
            return mean, self.log_std[intention_idx]

    def action_sample(self, mean: torch.Tensor, log_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates an action sample from the policy network. The output of the network is assumed to be gaussian
        distributed. Let u be a random variable with distribution p(u|s). Since we want our actions to be bound in
        [-1, 1] we apply the tanh function to u, that is a = tanh(u). By change of variable, we have:

        π(a|s) = p(u|s) |det(da/du)| ^-1. Since da/du = diag(1 - tanh^2(u)). We obtain the log likelihood as
        log π(a|s) = log p(u|s) - ∑_i 1 - tanh^2(u_i)

        Args:
            mean: μ(s)
            log_std: log σ(s) where u ~ N(•|μ(s), σ(s))

        Returns:
            action sample a = tanh(u) and log prob log π(a|s)
        """
        dist = self.normal_dist(mean, log_std)
        normal_action = dist.rsample()  # rsample() employs reparameterization trick
        action = torch.tanh(normal_action)
        normal_log_prob = dist.log_prob(normal_action)
        log_prob = torch.sum(normal_log_prob, dim=-1) - torch.sum(torch.log((1 - action.pow(2) + self.eps)), dim=-1)
        return action, log_prob

    def get_log_prob(self, actions: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor,
                     normal_actions: torch.Tensor = None) -> torch.Tensor:
        """
        Returns the log prob of a given action a = tanh(u) and u ~ N(•|μ(s), σ(s)) according to

        log π(a|s) = log p(u|s) - ∑_i 1 - tanh^2(u_i).

        If u is not given we can reconstruct it with u = tanh^-1(a), since tanh is bijective.

        Args:
            actions: a = tanh(u)
            mean: μ(s)
            log_std: log σ(s)
            normal_actions: u ~ N(•|μ(s), σ(s))

        Returns:
            log π(a|s)
        """
        if normal_actions is None:
            normal_actions = self.inverseTanh(actions)

        normal_log_probs = self.normal_dist(mean, log_std).log_prob(normal_actions)
        log_probs = torch.sum(normal_log_probs, dim=-1) - torch.sum(torch.log(1 - actions.pow(2) + self.eps), dim=-1)
        assert not torch.isnan(log_probs).any()
        return log_probs

    @staticmethod
    def normal_dist(mean: torch.Tensor, log_std: torch.Tensor) -> Normal:
        """
        Returns a normal distribution.

        Args:
            mean: μ(s)
            log_std: log σ(s) where u ~ N(•|μ(s), σ(s))

        Returns:
            N(u|μ(s), σ(s))
        """
        return Normal(loc=mean, scale=log_std.exp())

    def inverseTanh(self, action: torch.Tensor) -> torch.Tensor:
        """
        Computes the inverse of the tanh for the given action
        Args:
            action: a = tanh(u)

        Returns:
            u = tanh^-1(a)
        """
        eps = torch.finfo(action.dtype).eps  # The smallest representable number such that 1.0 + eps != 1.0
        atanh = self.atanh(action.clamp(min=-1. + eps, max=1. - eps))
        assert not torch.isnan(atanh).any()
        return atanh

    @staticmethod
    def atanh(action: torch.Tensor) -> torch.Tensor:
        return 0.5 * (action.log1p() - (-action).log1p())


class Critic(Base):
    def __init__(self,
                 num_intentions: int,
                 num_actions: int,
                 num_obs: int,
                 base_layer_dims: List = None,
                 intention_layer_dims: List = None,
                 non_linearity: nn.Module = nn.ReLU(),
                 logger=None):

        if base_layer_dims is None:
            base_layer_dims = [128, 128]
        if intention_layer_dims is None:
            intention_layer_dims = [64]

        super(Critic, self).__init__(base_layer_dims=[num_actions + num_obs] + base_layer_dims,
                                     non_linearity=non_linearity)

        self.num_intentions = num_intentions
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.logger = logger

        # Create a model for a intention net
        intention_modules = []
        for i in range(len(intention_layer_dims) - 1):
            intention_modules.append(nn.Linear(intention_layer_dims[i], intention_layer_dims[i + 1]))
            intention_modules.append(non_linearity)
        intention_modules.append(nn.Linear(intention_layer_dims[-1], 1))

        # Create all intention nets
        self.intention_nets = []
        for i in range(num_intentions):
            intention_net = nn.Sequential(*intention_modules[:-1])  # Remove last non-linearity
            self.init_weights(intention_net)
            self.intention_nets.append(intention_net)

    def __call__(self, actions, observations):
        assert actions.shape == (self.num_intentions, self.num_actions)
        assert observations.shape == self.num_obs
        x = torch.cat([actions, observations.expand(self.num_intentions, observations.shape[0])], dim=-1)
        return self.forward(x)

    def forward(self, x):
        x = self.base_model(x)

        Q_values = torch.tensor([self.num_intentions, 1], dtype=torch.float32)
        for i in range(self.num_intentions):
            Q_values[i, :] = self.intention_nets[i](x[i])

        return Q_values
