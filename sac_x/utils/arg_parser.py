from argparse import ArgumentParser


class ArgParser(ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__(description='arg_parser')

        # Algorithm arguments
        self.add_argument('--num_workers', type=int, default=2,
                          help='Number of workers training the agent in parallel.')
        self.add_argument('--num_grads', type=int, default=1,
                          help='Number of gradients collected before updating the networks.')
        self.add_argument('--update_targnets_every', type=int, default=10,
                          help='Number of learning steps before the target networks are updated.')
        self.add_argument('--learning_steps', type=int, default=20,
                          help='Total number of learning timesteps before sampling trajectories.')
        self.add_argument('--num_runs', type=int, default=5000,
                          help='Number of learning iterations.')
        self.add_argument('--actor_lr', type=float, default=2e-4,
                          help='Learning rate for the actor network.')
        self.add_argument('--critic_lr', type=float, default=2e-4,
                          help='Learning rate for the critic network.')
        self.add_argument('--global_gradient_norm', type=float, default=0.5,
                          help='Enables gradient clipping with a specified global parameter L2 norm')
        self.add_argument('--entropy_reg', type=float, default=0,
                          help='Scaling of entropy term in the actor loss function')
        self.add_argument('--replay_buffer_size', type=int, default=300,
                          help='Size of the replay buffer.')
        self.add_argument('--num_trajectories', type=int, default=10,
                          help='Number of trajectories sampled before entering the learning phase.')
        self.add_argument('--schedule_switch', type=int, default=180,
                          help='Number of time steps after the scheduler samples a new intention.')
        self.add_argument('--discount_factor', type=float, default=0.99,
                          help='Discount factor for future rewards.')

        # Environment arguments
        self.add_argument('--num_actions', type=int, default=3,
                          help='Dimension of the action space.')
        self.add_argument('--num_observations', type=int, default=29,
        # self.add_argument('--num_observations', type=int, default=15,
                          help='Dimension of the observation space.')
        self.add_argument('--num_intentions', type=int, default=10,
                          help='Number of intentions (auxiliary tasks + external tasks).')
        self.add_argument('--episode_length', type=int, default=360,
                          help='Number of steps the agent interacts with the environment.')
