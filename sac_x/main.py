import torch.multiprocessing as mp

from sac_x.utils.arg_parser import ArgParser

parser = ArgParser()


def work(param_server, replay_buffer, parser_args, condition):
    worker = Agent(param_server=param_server,
                   shared_replay_buffer=replay_buffer,
                   parser_args=parser_args,
                   condition=condition)
    worker.run()


def run_server(param_server):
    param_server.run()

if __name__ == '__main__':
    lock = mp.Lock()
    worker_cv = mp.Condition(lock)
    server_cv = mp.Condition(lock)
    args = parser.parse_args()
