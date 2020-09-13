import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import pathlib


class Evaluator:
    def __init__(self,
                 env,
                 actor: torch.nn.Module,
                 critic: torch.nn.Module,
                 parser_args,
                 logger: SummaryWriter = None):

        self.model_root_dir = str(pathlib.Path(__file__).resolve().parents[1]) + "/stack/"

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
        for intention_idx in range(self.num_intentions):
            rewards = []
            obs = torch.tensor(self.env.reset(), dtype=torch.float)
            for t in range(self.trajectory_length):
                mean, log_std = self.actor(obs, intention_idx)
                action, action_log_pr = self.actor.action_sample(mean, torch.zeros_like(mean))
                denormalized_action = action.detach().cpu().numpy() * self.env.action_space.high[:3]
                next_obs, reward, done, _ = self.env.step(denormalized_action)
                next_obs = torch.tensor(next_obs, dtype=torch.float)
                obs = next_obs

                rewards.append(reward[intention_idx])

            R[intention_idx] = sum([0.99**t * r for t, r in enumerate(rewards)])

        if R[9] != 0:  # Save the model if the stack reward is not zero
            torch.save(self.actor.state_dict(), self.model_root_dir + "/" + "actor_stack" + datetime.datetime.now().strftime("%d-%m_%H-%M"))
        #
        # if R[3] != 0:  # Save the model if the above reward is not zero
        #     torch.save(self.actor.state_dict(), self.model_root_dir + "/" + "actor_above" + datetime.datetime.now().strftime("%d-%m_%H-%M"))

        self.logger.log_rewards(R)

        #################

        # R = torch.zeros([self.num_intentions])
        # for intention_idx in range(self.num_intentions):
        #     rewards = []
        #     obs = torch.tensor(self.env.reset(), dtype=torch.float)
        #     for t in range(self.trajectory_length):
        #         mean, log_std = self.actor(obs, intention_idx)
        #         action, action_log_pr = self.actor.action_sample(mean, log_std)
        #         denormalized_action = action.detach().cpu().numpy() * self.env.action_space.high[:3]
        #         next_obs, reward, done, _ = self.env.step(denormalized_action)
        #         next_obs = torch.tensor(next_obs, dtype=torch.float)
        #         obs = next_obs
        #
        #         rewards.append(reward[intention_idx])
        #
        #     R[intention_idx] = sum([0.99 ** t * r for t, r in enumerate(rewards)])
        #
        # # if R[-1] != 0:  # Save the model if the stack reward is not zero
        # #     torch.save(self.actor.state_dict(),
        # #                self.model_root_dir + "/" + "actor_stack" + datetime.datetime.now().strftime("%d-%m_%H-%M"))
        #
        # self.logger.log_rewards(R, mode="Eval Std")
