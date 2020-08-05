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
            obs = torch.tensor(self.env.reset(), dtype=torch.float)
            for t in range(self.trajectory_length):
                mean, log_std = self.actor(obs, intention_idx)
                action, action_log_pr = self.actor.action_sample(mean, torch.zeros_like(mean) * -1e10)
                next_obs, reward, done, _ = self.env.step(action.detach().cpu())
                next_obs = torch.tensor(next_obs, dtype=torch.float)
                obs = next_obs

                rewards.append(reward)

            R[intention_idx] = np.mean(rewards)

        self.logger.log_rewards(R)


