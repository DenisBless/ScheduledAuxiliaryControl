import torch
import torch.nn as nn
from typing import List
import numpy as np


class Base(nn.Module):
    def __init__(self,
                 base_layer_dims: List,
                 non_linearity: nn.Module = nn.ReLU):
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
                 non_linearity: nn.Module = nn.ReLU,
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

    def forward(self, x, intention_idx=None):
        assert 0 <= intention_idx < self.num_intentions
        assert self.log_std[intention_idx].shape == [self.num_actions]

        x = self.base_model(x)

        if intention_idx is None:
            means = torch.FloatTensor([self.num_intentions, self.num_actions])
            for i in range(self.num_intentions):
                means[i, :] = self.intention_nets[i](x)
            return means, self.log_std

        else:
            mean = self.intention_nets[intention_idx](x)
            return mean, self.log_std[intention_idx]
        #
        # dist = torch.distributions.Normal(loc=mean, scale=self.log_std[intention_idx])
        # raw_action = dist.rsample()  # rsample() enables reparameterization trick
        # action = torch.tanh(raw_action)
        # log_prob = dist.log_prob(raw_action).sum(dim=-1) - torch.sum(torch.log((1 - action.pow(2) + self.eps)), dim=-1)
        # return action, log_prob


class Critic(Base):
    def __init__(self,
                 num_intentions: int,
                 num_actions: int,
                 num_obs: int,
                 base_layer_dims: List = None,
                 intention_layer_dims: List = None,
                 non_linearity: nn.Module = nn.ReLU,
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

    def forward(self, actions, observations):
        assert actions.shape == (self.num_intentions, self.num_actions)
        assert observations.shape == self.num_obs
        x = torch.cat([actions, observations.expand(self.num_intentions, observations.shape[0])], dim=-1)
        x = self.base_model(x)

        Q_values = torch.FloatTensor([self.num_intentions, 1])
        for i in range(self.num_intentions):
            Q_values[i, :] = self.intention_nets[i](x[i])

        return Q_values


