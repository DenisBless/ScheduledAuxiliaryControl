from argparse import ArgumentParser


class ArgParser(ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__(description='arg_parser')

        # Algorithm parameter
        self.add_argument('--num_worker', type=int, default=1,
                          help='Number of workers training the agent in parallel.')
        self.add_argument('--num_grads', type=int, default=1,
                          help='Number of gradients collected before updating the networks.')
        self.add_argument('--update_targnets_every', type=int, default=10,
                          help='Number of learning steps before the target networks are updated.')
        self.add_argument('--learning_steps', type=int, default=200,
                          help='Total number of learning timesteps before sampling trajectories.')
        self.add_argument('--num_runs', type=int, default=5000,
                          help='Number of learning iterations.')
        self.add_argument('--actor_lr', type=float, default=2e-4,
                          help='Learning rate for the actor network.')
        self.add_argument('--critic_lr', type=float, default=2e-4,
                          help='Learning rate for the critic network.')
        self.add_argument('--init_std', type=float, default=0.2,
                          help='Initial standard deviation of the actor.')
        self.add_argument('--smoothing_coefficient', type=float, default=1,
                          help='Decides how the target networks are updated. One corresponds to a hard updates, whereas'
                               ' values between zero and one result in exponential moving average updates.')
        self.add_argument('--global_gradient_norm', type=float, default=0.5,
                          help='Enables gradient clipping with a specified global parameter L2 norm')
        self.add_argument('--entropy_reg', type=float, default=0,
                          help='Scaling of entropy term in the actor loss function')
        self.add_argument('--replay_buffer_size', type=int, default=300,
                          help='Size of the replay buffer.')
        self.add_argument('--num_trajectories', type=int, default=20,
                          help='Number of trajectories sampled before entering the learning phase.')

    def hparam_dict(self):
        return {'update_targets': ...,
                'learning_steps': ...,
                'actor_lr': ...,
                'critic_lr': ...,
                'entropy_reg': ...,
                'init_std': ...,
                'global_gradient_norm': ...,
                'replay_buffer_size': ...,
                'num_trajectories': ...
                }

