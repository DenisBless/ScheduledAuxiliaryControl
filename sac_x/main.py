import torch.multiprocessing as mp

from sac_x.utils.arg_parser import ArgParser
from sac_x.sampler import Sampler
from sac_x.learner import Learner
from sac_x.parameter_server import ParameterServer
from sac_x.replay_buffer import SharedReplayBuffer
from sac_x.scheduler import SacU

arg_parser = ArgParser()


class Agent:
    def __init__(self, param_server, replay_buffer, scheduler, condition, parser_args):
        self.env = ...
        self.sampler = Sampler()
        self.learner = Learner()
        self.num_runs = parser_args.num_runs

    def run(self):
        for _ in range(self.num_runs):
            self.sampler.run()
            self.learner.run()


def work(param_server, replay_buffer, scheduler,  parser_args, condition):
    worker = Agent(param_server=param_server,
                   replay_buffer=replay_buffer,
                   scheduler=scheduler,
                   parser_args=parser_args,
                   condition=condition)
    worker.run()


def run_server(param_server):
    param_server.run()


if __name__ == '__main__':
    lock = mp.Lock()
    worker_cv = mp.Condition(lock)
    server_cv = mp.Condition(lock)
    p_args = arg_parser.parse_args()

    shared_param_server = ParameterServer(parser_args=p_args,
                                          worker_cv=worker_cv,
                                          server_cv=server_cv)

    shared_replay_buffer = SharedReplayBuffer(parser_args=p_args,
                                              cv=worker_cv)

    shared_scheduler = SacU(parser_args=p_args)
