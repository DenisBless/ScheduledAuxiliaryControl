import torch
from abc import ABC, abstractmethod


class Scheduler(metaclass=ABC):
    def __init__(self, num_intentions):
        self.num_intentions = num_intentions

    @abstractmethod
    def sample_intention(self) -> torch.Tensor:
        """
        Implements a sampler for the scheduler.

        Returns:
            Index of the sampled intention.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self) -> None:
        """
        Updates the scheduler according to its update rule.
        """
        raise NotImplementedError


class SacU(Scheduler):
    def __init__(self, num_intentions):
        super(SacU, self).__init__(num_intentions)

    def sample_intention(self) -> torch.Tensor:
        """
        Uniform sampler
        """
        return torch.randint(0, self.num_intentions, (1,))

    def update(self) -> None:
        pass


class SacQ(Scheduler):
    def __init__(self, num_intentions, M):
        super(SacQ, self).__init__(num_intentions)
        self.num_intentions = num_intentions
        self.M = M
        self.Q_table = ...

    def sample_intention(self) -> torch.Tensor:
        ...