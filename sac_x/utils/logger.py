from torch.utils.tensorboard import SummaryWriter


class Logger(SummaryWriter):
    def __init__(self, log_dir=None):
        super().__init__(log_dir=log_dir)

    def log_rewards(self, reward_vector, mode: str = 'Eval'):
        # self.add_scalar(tag=mode + ' rewards/close', scalar_value=reward_vector[0].mean())
        # self.add_scalar(tag=mode + ' rewards/above', scalar_value=reward_vector[1].mean())
        # self.add_scalar(tag=mode + ' rewards/above_close', scalar_value=reward_vector[2].mean())
        # self.add_scalar(tag=mode + ' rewards/left', scalar_value=reward_vector[3].mean())
        # self.add_scalar(tag=mode + ' rewards/left_close', scalar_value=reward_vector[4].mean())
        # self.add_scalar(tag=mode + ' rewards/below', scalar_value=reward_vector[5].mean())
        # self.add_scalar(tag=mode + ' rewards/right', scalar_value=reward_vector[6].mean())
        # self.add_scalar(tag=mode + ' rewards/below_close', scalar_value=reward_vector[7].mean())
        # self.add_scalar(tag=mode + ' rewards/right_close', scalar_value=reward_vector[8].mean())
        # self.add_scalar(tag=mode + ' rewards/move1', scalar_value=reward_vector[9].mean())
        # self.add_scalar(tag=mode + ' rewards/move2', scalar_value=reward_vector[10].mean())
        # self.add_scalar(tag=mode + ' rewards/touch', scalar_value=reward_vector[11].mean())
        # self.add_scalar(tag=mode + ' rewards/no_touch', scalar_value=reward_vector[12].mean())
        # self.add_scalar(tag=mode + ' rewards/stack', scalar_value=reward_vector[13].mean())

        # self.add_scalar(tag=mode + ' rewards/reach', scalar_value=reward_vector[0].mean())
        # self.add_scalar(tag=mode + ' rewards/close', scalar_value=reward_vector[1].mean())
        # self.add_scalar(tag=mode + ' rewards/above', scalar_value=reward_vector[2].mean())
        # self.add_scalar(tag=mode + ' rewards/above_close', scalar_value=reward_vector[3].mean())
        # self.add_scalar(tag=mode + ' rewards/left', scalar_value=reward_vector[4].mean())
        # self.add_scalar(tag=mode + ' rewards/move1', scalar_value=reward_vector[5].mean())
        # self.add_scalar(tag=mode + ' rewards/move2', scalar_value=reward_vector[6].mean())
        # self.add_scalar(tag=mode + ' rewards/touch', scalar_value=reward_vector[7].mean())
        # self.add_scalar(tag=mode + ' rewards/no_touch', scalar_value=reward_vector[8].mean())
        # self.add_scalar(tag=mode + ' rewards/stack', scalar_value=reward_vector[9].mean())

        self.add_scalar(tag=mode + ' rewards/reach', scalar_value=reward_vector[0].mean())
        self.add_scalar(tag=mode + ' rewards/above', scalar_value=reward_vector[1].mean())
        self.add_scalar(tag=mode + ' rewards/stack', scalar_value=reward_vector[2].mean())

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

    def log_std(self, std):
        # if std.shape[0] == 1:
        #     self.add_scalar(tag='Std/touch', scalar_value=std.mean())
        # else:
        # self.add_scalar(tag='Std/reach', scalar_value=std[0].mean())
        # self.add_scalar(tag='Std/close', scalar_value=std[1].mean())
        # self.add_scalar(tag='Std/above', scalar_value=std[2].mean())
        # self.add_scalar(tag='Std/above_close', scalar_value=std[3].mean())
        # self.add_scalar(tag='Std/left', scalar_value=std[4].mean())
        # self.add_scalar(tag='Std/move1', scalar_value=std[5].mean())
        # self.add_scalar(tag='Std/move2', scalar_value=std[6].mean())
        # self.add_scalar(tag='Std/touch', scalar_value=std[7].mean())
        # self.add_scalar(tag='Std/no_touch', scalar_value=std[8].mean())
        # self.add_scalar(tag='Std/stack', scalar_value=std[9].mean())
        #
        # self.add_scalar(tag='Std std/reach', scalar_value=std[0].std())
        # self.add_scalar(tag='Std std/close', scalar_value=std[1].std())
        # self.add_scalar(tag='Std std/above', scalar_value=std[2].std())
        # self.add_scalar(tag='Std std/above_close', scalar_value=std[3].std())
        # self.add_scalar(tag='Std std/left', scalar_value=std[4].std())
        # self.add_scalar(tag='Std std/move1', scalar_value=std[5].std())
        # self.add_scalar(tag='Std std/move2', scalar_value=std[6].std())
        # self.add_scalar(tag='Std std/touch', scalar_value=std[7].std())
        # self.add_scalar(tag='Std std/no_touch', scalar_value=std[8].std())
        # self.add_scalar(tag='Std std/stack', scalar_value=std[9].std())

        # self.add_histogram(tag='Std std/reach', values=std[0])
        # self.add_histogram(tag='Std std/close', values=std[1])
        # self.add_histogram(tag='Std std/above', values=std[2])
        # self.add_histogram(tag='Std std/above_close', values=std[3])
        # self.add_histogram(tag='Std std/left', values=std[4])
        # self.add_histogram(tag='Std std/move1', values=std[5])
        # self.add_histogram(tag='Std std/move2', values=std[6])
        # self.add_histogram(tag='Std std/touch', values=std[7])
        # self.add_histogram(tag='Std std/no_touch', values=std[8])
        # self.add_histogram(tag='Std std/stack', values=std[9])
        #
        self.add_histogram(tag='Std std/reach', values=std[0])
        self.add_histogram(tag='Std std/above', values=std[1])
        self.add_histogram(tag='Std std/stack', values=std[2])

    def log_schedule_decisions(self, schedule_decisions):
        for i, d in enumerate(schedule_decisions):
            self.add_scalar(tag='Scheduler/decision' + str(i), scalar_value=d)
