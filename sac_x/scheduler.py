import torch
import torch.nn.functional as F
from torch.distributions.multinomial import Multinomial
from abc import ABC, abstractmethod


class Scheduler(ABC):
    def __init__(self, num_intentions):
        self.num_intentions = num_intentions

    @abstractmethod
    def sample_intention(self, tasks, schedule_decisions=None) -> torch.Tensor:
        """
        Implements a sampler for the scheduler.

        Returns:
            Index of the sampled intention.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, R_main, tasks) -> None:
        """
        Updates the scheduler according to its update rule.
        """
        raise NotImplementedError


class SacU(Scheduler):
    def __init__(self, parser_args):
        super(SacU, self).__init__(parser_args.num_intentions)

    def sample_intention(self, tasks, schedule_decision=None) -> torch.Tensor:
        """
        Uniform sampler
        """
        return torch.randint(0, self.num_intentions, (1,))

    def update(self, R_main, tasks) -> None:
        pass


class SacQ(Scheduler):
    def __init__(self, parser_args, M=50,  H=2, temperature=1):
        super(SacQ, self).__init__(parser_args.num_intentions)
        num_intentions = parser_args.num_intentions
        self.schedule_switch = parser_args.schedule_switch
        self.H = H  # number of different tasks per episode
        self.M = M  # number of trajectories for MC estimate
        self.M_task = torch.zeros([self.H, num_intentions])
        self.temperature = temperature
        self.Q_table = {
            0: torch.zeros([num_intentions]),
            1: torch.zeros([num_intentions, num_intentions])
        }

    def sample_intention(self, h, schedule_decisions=None) -> torch.Tensor:
        if h == 0:
            Ps = F.softmax(self.Q_table[0] / self.temperature)
        elif h == 1:
            Ps = F.softmax((self.Q_table[1][schedule_decisions[0]] / self.temperature))
        else:
            raise ValueError("Invalid number of tasks per episode.")

        # return int(Multinomial(probs=Ps).sample().item())
        return torch.where(Multinomial(probs=Ps).sample() == 1)[0]

    def update(self, R_main, tasks) -> None:
        # Update scheduler for h = 0
        self.M_task[0, tasks[0]] += 1
        delta_0 = sum(R_main[:self.schedule_switch]) - self.Q_table[0][tasks[0]]
        self.Q_table[0][tasks[0]] += delta_0 / self.M  # self.M_task[0, tasks[0]]

        # Update scheduler for h = 1
        self.M_task[1, tasks[1]] += 1
        delta_1 = sum(R_main[self.schedule_switch:]) - self.Q_table[1][tasks[0], tasks[1]]
        self.Q_table[1][tasks[0], tasks[1]] += delta_1 / self.M  # self.M_task[1, tasks[1]]

        if delta_0 or delta_1:
            print(self.Q_table)
