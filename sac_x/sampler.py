import numpy as np
import torch
from torch.multiprocessing import current_process
from torch.utils.tensorboard import SummaryWriter

from sac_x.replay_buffer import SharedReplayBuffer
from sac_x.scheduler import Scheduler


class Sampler:
    def __init__(self,
                 env,
                 actor: torch.nn.Module,
                 replay_buffer: SharedReplayBuffer,
                 scheduler: Scheduler,
                 argp):

        self.env = env
        self.actor = actor
        self.replay_buffer = replay_buffer
        self.scheduler = scheduler
        self.num_trajectories = argp.num_trajectories
        self.trajectory_length = argp.episode_length
        self.schedule_switch = argp.schedule_switch
        self.discount_factor = argp.discount_factor

        self.discounts = torch.cumprod(torch.ones([self.trajectory_length - 1]) * 0.99, dim=-1)

        self.log_every = 10

    def run(self) -> None:
        for i in range(self.num_trajectories):
            states, actions, rewards, action_log_prs, schedule_decisions = [], [], [], [], []
            h = 0
            intention_idx = None
            obs = torch.tensor(self.env.reset(), dtype=torch.float)
            for t in range(self.trajectory_length):

                # Sample an intention from the scheduler
                if t % self.schedule_switch == 0:
                    intention_idx = self.scheduler.sample_intention(h)
                    schedule_decisions.append(intention_idx[0])
                    h += 1

                mean, log_std = self.actor(obs, intention_idx)
                action, action_log_pr = self.actor.action_sample(mean, log_std)
                denormalized_action = action.detach().cpu().numpy() * self.env.action_space.high
                assert self.env.action_space.low.all() <= denormalized_action.all() <= self.env.action_space.high.all()
                next_obs, reward, done, _ = self.env.step(denormalized_action)
                next_obs = torch.tensor(next_obs, dtype=torch.float)
                reward = torch.tensor(reward, dtype=torch.float)
                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                action_log_prs.append(action_log_pr)
                obs = next_obs

            # turn lists into tensors
            states = torch.stack(states)
            actions = torch.stack(actions)
            rewards = torch.stack(rewards)
            action_log_prs = torch.stack(action_log_prs)
            schedule_decisions = torch.stack(schedule_decisions)

            # main_cum_reward = rewards[0, 13] + (rewards[1:, 13] * self.discounts).sum()
            self.scheduler.update(None, schedule_decisions)  # Update the scheduler
            self.replay_buffer.push(states.detach(), actions.detach(), rewards.detach(), action_log_prs.detach(),
                                    schedule_decisions)
