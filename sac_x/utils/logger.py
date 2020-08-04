from torch.utils.tensorboard import SummaryWriter


class Logger(SummaryWriter):
    def __init__(self):
        super().__init__()

    def log_rewards(self, auxiliary, external_task):
        self.add_scalar(tag='auxiliary_rewards/close', scalar_value=auxiliary[...])
        self.add_scalar(tag='auxiliary_rewards/above', scalar_value=auxiliary[...])
        self.add_scalar(tag='auxiliary_rewards/above_close', scalar_value=auxiliary[...])
        self.add_scalar(tag='auxiliary_rewards/left', scalar_value=auxiliary[...])
        self.add_scalar(tag='auxiliary_rewards/left_close', scalar_value=auxiliary[...])
        self.add_scalar(tag='auxiliary_rewards/below', scalar_value=auxiliary[...])
        self.add_scalar(tag='auxiliary_rewards/right', scalar_value=auxiliary[...])
        self.add_scalar(tag='auxiliary_rewards/below_close', scalar_value=auxiliary[...])
        self.add_scalar(tag='auxiliary_rewards/right_close', scalar_value=auxiliary[...])
        self.add_scalar(tag='auxiliary_rewards/move', scalar_value=auxiliary[...])
        self.add_scalar(tag='auxiliary_rewards/touch', scalar_value=auxiliary[...])
        self.add_scalar(tag='auxiliary_rewards/no_touch', scalar_value=auxiliary[...])

        self.add_scalar(tag='external_rewards/stack', scalar_value=external_task[...])
