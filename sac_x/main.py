import time
import torch.multiprocessing as mp

from sac_x.utils.arg_parser import ArgParser
from sac_x.utils.logger import Logger
from sac_x.utils.model_saver import ModelSaver
from sac_x.sampler import Sampler
from sac_x.learner import Learner
from sac_x.evaluator import Evaluator
from sac_x.parameter_server import ParameterServer
from sac_x.replay_buffer import SharedReplayBuffer
from sac_x.scheduler import SacU
from sac_x.actor_critic_nets import Actor, Critic

from simulation.src.gym_sf.mujoco.mujoco_envs.stack_env.stack_env import StackEnv

arg_parser = ArgParser()


class Agent:
    def __init__(self, param_server, replay_buffer, scheduler, parser_args):
        self.process_id = mp.current_process()._identity[0]  # process ID

        self.model_saver = ModelSaver() if self.process_id == 1 else None
        logger = Logger() if self.process_id == 1 else None

        with param_server.worker_cv:
            env = StackEnv(max_steps=parser_args.episode_length, control_timesteps=5, percentage=0.02, dt=1e-2,
                           render=False)

        actor = Actor(parser_args=parser_args)
        critic = Critic(parser_args=parser_args)

        self.sampler = Sampler(env=env, actor=actor, replay_buffer=replay_buffer, scheduler=scheduler,
                               argp=parser_args, logger=logger)
        self.learner = Learner(actor=actor, critic=critic, parameter_server=param_server,
                               replay_buffer=replay_buffer, parser_args=parser_args, logger=logger)

        self.evaluator = Evaluator(env=env, actor=actor, critic=critic, parser_args=parser_args, logger=logger)

        self.num_runs = parser_args.num_runs

    def run(self):
        for i in range(self.num_runs):
            if self.process_id == 1:
                t1 = time.time()
                self.sampler.run()
                t2 = time.time()
                print("Sampling Nr.", i + 1, " finished. Time taken: ", t2 - t1)
                self.learner.run()
                print("Learning Nr.", i + 1, " finished. Time taken: ", time.time() - t1)
                self.evaluator.run()
                self.model_saver.save_model(self.learner.parameter_server.shared_actor, 'actor')

            else:
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


def sample(param_server, replay_buffer, scheduler, parser_args):
    # Initialize environment
    with param_server.worker_cv:
        env = StackEnv(max_steps=parser_args.episode_length, control_timesteps=5, percentage=0.02, dt=1e-2)

    actor = Actor(parser_args=parser_args)
    sampler = Sampler(env=env, actor=actor, replay_buffer=replay_buffer, scheduler=scheduler, argp=parser_args)

    while True:  # Sample until learners are finished
        sampler.run()


def learn(param_server, replay_buffer, scheduler, parser_args):
    process_id = mp.current_process()._identity[0]  # process ID

    model_saver = ModelSaver() if process_id == 1 else None
    logger = Logger() if process_id == 1 else None

    actor = Actor(parser_args=parser_args)
    critic = Critic(parser_args=parser_args)

    learner = Learner(actor=actor, critic=critic, parameter_server=param_server,
                      replay_buffer=replay_buffer, parser_args=parser_args, logger=logger)

    for i in range(parser_args.num_runs):
        learner.run()
        model_saver.save_model(param_server.shared_actor, 'actor')

    exit()


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
