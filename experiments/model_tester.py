import torch
import pathlib

from sac_x.actor_critic_nets import Actor
from sac_x.utils.arg_parser import ArgParser
from simulation.src.gym_sf.mujoco.mujoco_envs.stack_env.stack_env import StackEnv

arg_parser = ArgParser()


class ModelTester:
    def __init__(self,
                 env,
                 actor: torch.nn.Module,
                 intention_idx: int,
                 parser_args):

        self.env = env
        self.actor = actor
        self.intention_idx = intention_idx
        self.num_actions = parser_args.num_actions
        self.num_observations = parser_args.num_observations
        self.num_intentions = parser_args.num_intentions
        self.num_trajectories = parser_args.num_trajectories
        self.trajectory_length = parser_args.episode_length

    def run(self) -> None:
        while True:
            intention = int(input("Select Intention: \n => "))
            assert 0 <= intention <= self.num_intentions, "Error, invalid intention index."

            for _ in range(3):
                obs = torch.tensor(self.env.reset(), dtype=torch.float)
                for t in range(self.trajectory_length):
                    # mean, log_std = self.actor(obs, self.intention_idx)
                    mean, log_std = self.actor(obs, intention)
                    action, action_log_pr = self.actor.action_sample(mean, torch.ones_like(mean) * -1e10)
                    denormalized_action = action.detach().cpu().numpy() * self.env.action_space.high
                    assert self.env.action_space.low.all() <= denormalized_action.all() <= self.env.action_space.high.all()
                    next_obs, reward, done, _ = self.env.step(denormalized_action)
                    # print(reward)
                    next_obs = torch.tensor(next_obs, dtype=torch.float)
                    obs = next_obs


if __name__ == '__main__':
    INTENTION_IDX = 0
    PATH_TO_MODEL = str(pathlib.Path(__file__).resolve().parents[1]) + "/models/" + "11-08_21-40/actor_250"

    parser_args = arg_parser.parse_args()
    actor = Actor(parser_args=parser_args)
    actor.load_state_dict(torch.load(PATH_TO_MODEL))

    env = StackEnv(max_steps=parser_args.episode_length, control_timesteps=5, percentage=0.012, dt=1e-2, render=True)
    model_tester = ModelTester(env=env, actor=actor, intention_idx=INTENTION_IDX, parser_args=parser_args)

    model_tester.run()
