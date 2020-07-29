import torch
import torch.nn as nn
from typing import List
import numpy as np


class Base(nn.Module):
    def __init__(self,
                 base_layer_dims: List,
                 non_linearity: nn.Module = nn.ReLU):
        super(Base, self).__init__()

        actor_base_modules = []
        for i in range(len(base_layer_dims) - 1):
            actor_base_modules.append(torch.nn.Linear(base_layer_dims[i], base_layer_dims[i + 1]))
            actor_base_modules.append(non_linearity)

        self.actor_base_model = nn.Sequential(*actor_base_modules)
        self.init_weights(self.actor_base_model)

    def forward(self, x):
        return self.actor_base_model(x)

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

    def forward(self, x, intention_idx=None):
        assert 0 <= intention_idx < self.num_intentions
        x = self.actor_base_model(x)
        if intention_idx is None:
            ...
        else:
            x = self.intention_nets[intention_idx](x)


