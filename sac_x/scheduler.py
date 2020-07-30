import torch
from abc import ABC, abstractmethod
from torch.distributions.uniform import Uniform


class Base(metaclass=ABC):
    def __init__(self, num_intentions):
        self.num_intentions = num_intentions

    @abstractmethod
    def sample_intention(self) -> int:
        ...

    @abstractmethod
    def update(self) -> None:
        ...


class SacU(Base):
    def __init__(self, num_intentions):
        super(SacU, self).__init__(num_intentions)

    def sample_intention(self) -> torch.Tensor:
        """
        Uniform sampler.

        Returns:
            Intention index
        """
        return torch.randint(0, self.num_intentions, (1,))

    def update(self) -> None:
        pass



class SacQ(Base):
    def __init__(self, num_intentions):
        super(SacQ, self).__init__(num_intentions)
        self.Q_table = ...

