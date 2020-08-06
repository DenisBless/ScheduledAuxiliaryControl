import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class Evaluator:
    def __init__(self,
                 env,
                 actor: torch.nn.Module,
                 critic: torch.nn.Module,
                 parser_args,
                 logger: SummaryWriter = None):

        self.env = env
        self.actor = actor
        self.critic = critic
        self.num_actions = parser_args.num_actions
        self.num_observations = parser_args.num_observations
        self.num_intentions = parser_args.num_intentions
        self.num_trajectories = parser_args.num_trajectories
        self.trajectory_length = parser_args.episode_length

        self.logger = logger

    def run(self) -> None:
        R = torch.zeros([self.num_intentions])
        # Q = torch.zeros([self.num_intentions, self.trajectory_length])
        # A = torch.zeros([self.num_intentions, self.trajectory_length, self.num_actions])
        # S = torch.zeros([self.num_intentions, self.trajectory_length, self.num_observations])
        for intention_idx in range(self.num_intentions):
            rewards = []
            obs = torch.tensor(self.env.reset(), dtype=torch.float).to('cuda:0')
            for t in range(self.trajectory_length):
                mean, log_std = self.actor(obs, intention_idx)
                mean = mean.to('cuda:0')
                log_std = log_std.to('cuda:0')
                action, action_log_pr = self.actor.action_sample(mean, torch.ones_like(mean) * -1e10)
                denormalized_action = action.detach().cpu().numpy() * self.env.action_space.high
                assert self.env.action_space.low.all() <= denormalized_action.all() <= self.env.action_space.high.all()
                next_obs, reward, done, _ = self.env.step(denormalized_action)
                next_obs = torch.tensor(next_obs, dtype=torch.float).to('cuda:0')
                obs = next_obs

                rewards.append(reward[intention_idx])

            R[intention_idx] = np.mean(rewards)

        self.logger.log_rewards(R)
