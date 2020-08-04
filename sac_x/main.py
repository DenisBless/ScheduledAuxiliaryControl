import torch.multiprocessing as mp

from sac_x.utils.arg_parser import ArgParser
from sac_x.sampler import Sampler
from sac_x.learner import Learner
from sac_x.parameter_server import ParameterServer
from sac_x.replay_buffer import SharedReplayBuffer
from sac_x.scheduler import SacU
from sac_x.actor_critic_nets import Actor, Critic

from simulation.src.gym_sf.mujoco.mujoco_envs.stack_env.stack_env import StackEnv

arg_parser = ArgParser()


class Agent:
    def __init__(self, param_server, replay_buffer, scheduler, parser_args):
        env = StackEnv(max_steps=parser_args.episode_length, control_timesteps=5, percentage=0.015, dt=1e-2)

        actor = Actor(num_intentions=parser_args.num_intentions,
                      num_actions=parser_args.num_actions,
                      num_obs=parser_args.num_obs,
                      std_init=parser_args.std_init)

        critic = Critic(num_intentions=parser_args.num_intentions,
                        num_actions=parser_args.num_actions,
                        num_obs=parser_args.num_obs)

        self.sampler = Sampler(env=env, actor=actor, replay_buffer=replay_buffer, scheduler=scheduler, argp=parser_args)
        self.learner = Learner(actor=actor, critic=critic, parameter_server=param_server,
                               replay_buffer=replay_buffer, parser_args=parser_args)

        self.num_runs = parser_args.num_runs

    def run(self):
        for _ in range(self.num_runs):
            self.sampler.run()
            self.learner.run()


def work(param_server, replay_buffer, scheduler, parser_args):
    worker = Agent(param_server=param_server,
                   replay_buffer=replay_buffer,
                   scheduler=scheduler,
                   parser_args=parser_args)
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

    if p_args.num_workers > 1:
        processes = [mp.Process(target=work, args=(shared_param_server, shared_replay_buffer, shared_scheduler, p_args))
                     for _ in range(p_args.num_workers)]

        # Add a process for the  parameter server
        processes.append(mp.Process(target=run_server, args=(shared_param_server,)))

        for p in processes:
            p.start()

        for p in processes:
            p.join()

    else:
        raise ValueError("Error, the number of workers has to be > 1.")
