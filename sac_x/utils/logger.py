from torch.utils.tensorboard import SummaryWriter


class Logger(SummaryWriter):
    def __init__(self, log_dir=None):
        super().__init__(log_dir=log_dir)

    def log_rewards(self, reward_vector):
        if reward_vector.shape[0] == 1:
            self.add_scalar(tag='Rewards/touch', scalar_value=reward_vector.mean())
        else:
            self.add_scalar(tag='Rewards/close', scalar_value=reward_vector[0].mean())
            self.add_scalar(tag='Rewards/above', scalar_value=reward_vector[1].mean())
            self.add_scalar(tag='Rewards/above_close', scalar_value=reward_vector[2].mean())
            self.add_scalar(tag='Rewards/left', scalar_value=reward_vector[3].mean())
            self.add_scalar(tag='Rewards/left_close', scalar_value=reward_vector[4].mean())
            self.add_scalar(tag='Rewards/below', scalar_value=reward_vector[5].mean())
            self.add_scalar(tag='Rewards/right', scalar_value=reward_vector[6].mean())
            self.add_scalar(tag='Rewards/below_close', scalar_value=reward_vector[7].mean())
            self.add_scalar(tag='Rewards/right_close', scalar_value=reward_vector[8].mean())
            self.add_scalar(tag='Rewards/move1', scalar_value=reward_vector[9].mean())
            self.add_scalar(tag='Rewards/move2', scalar_value=reward_vector[10].mean())
            self.add_scalar(tag='Rewards/touch', scalar_value=reward_vector[11].mean())
            self.add_scalar(tag='Rewards/no_touch', scalar_value=reward_vector[12].mean())
            self.add_scalar(tag='Rewards/stack', scalar_value=reward_vector[13].mean())
            
    def log_Q_values(self, Q_values):
        if Q_values.shape[0] == 1:
            self.add_scalar(tag='Q_values/touch', scalar_value=Q_values.mean())
        else:
            self.add_scalar(tag='Q_values/close', scalar_value=Q_values[0].mean())
            self.add_scalar(tag='Q_values/above', scalar_value=Q_values[1].mean())
            self.add_scalar(tag='Q_values/above_close', scalar_value=Q_values[2].mean())
            self.add_scalar(tag='Q_values/left', scalar_value=Q_values[3].mean())
            self.add_scalar(tag='Q_values/left_close', scalar_value=Q_values[4].mean())
            self.add_scalar(tag='Q_values/below', scalar_value=Q_values[5].mean())
            self.add_scalar(tag='Q_values/right', scalar_value=Q_values[6].mean())
            self.add_scalar(tag='Q_values/below_close', scalar_value=Q_values[7].mean())
            self.add_scalar(tag='Q_values/right_close', scalar_value=Q_values[8].mean())
            self.add_scalar(tag='Q_values/move1', scalar_value=Q_values[9].mean())
            self.add_scalar(tag='Q_values/move2', scalar_value=Q_values[10].mean())
            self.add_scalar(tag='Q_values/touch', scalar_value=Q_values[11].mean())
            self.add_scalar(tag='Q_values/no_touch', scalar_value=Q_values[12].mean())
            self.add_scalar(tag='Q_values/stack', scalar_value=Q_values[13].mean())
