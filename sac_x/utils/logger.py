import torch
from torch.utils.tensorboard import SummaryWriter


class Logger(SummaryWriter):
    def __init__(self, log_dir=None):
        super().__init__(log_dir=log_dir)

    def log_rewards(self, reward_vector, mode='Eval'):

        self.add_scalar(tag=mode + 'rewards/close', scalar_value=reward_vector[0].mean())
        self.add_scalar(tag=mode + 'rewards/above', scalar_value=reward_vector[1].mean())
        self.add_scalar(tag=mode + 'rewards/above_close', scalar_value=reward_vector[2].mean())
        self.add_scalar(tag=mode + 'rewards/left', scalar_value=reward_vector[3].mean())
        self.add_scalar(tag=mode + 'rewards/left_close', scalar_value=reward_vector[4].mean())
        self.add_scalar(tag=mode + 'rewards/move1', scalar_value=reward_vector[5].mean())
        self.add_scalar(tag=mode + 'rewards/move2', scalar_value=reward_vector[6].mean())
        self.add_scalar(tag=mode + 'rewards/touch', scalar_value=reward_vector[7].mean())
        self.add_scalar(tag=mode + 'rewards/no_touch', scalar_value=reward_vector[8].mean())
        self.add_scalar(tag=mode + 'rewards/stack', scalar_value=reward_vector[9].mean())
        # self.add_scalar(tag=mode + 'rewards/close', scalar_value=reward_vector[0].mean())
        # self.add_scalar(tag=mode + 'rewards/above', scalar_value=reward_vector[1].mean())
        # self.add_scalar(tag=mode + 'rewards/above_close', scalar_value=reward_vector[2].mean())
        # self.add_scalar(tag=mode + 'rewards/left', scalar_value=reward_vector[3].mean())
        # self.add_scalar(tag=mode + 'rewards/left_close', scalar_value=reward_vector[4].mean())
        # self.add_scalar(tag=mode + 'rewards/below', scalar_value=reward_vector[5].mean())
        # self.add_scalar(tag=mode + 'rewards/right', scalar_value=reward_vector[6].mean())
        # self.add_scalar(tag=mode + 'rewards/below_close', scalar_value=reward_vector[7].mean())
        # self.add_scalar(tag=mode + 'rewards/right_close', scalar_value=reward_vector[8].mean())
        # self.add_scalar(tag=mode + 'rewards/move1', scalar_value=reward_vector[9].mean())
        # self.add_scalar(tag=mode + 'rewards/move2', scalar_value=reward_vector[10].mean())
        # self.add_scalar(tag=mode + 'rewards/touch', scalar_value=reward_vector[11].mean())
        # self.add_scalar(tag=mode + 'rewards/no_touch', scalar_value=reward_vector[12].mean())
        # self.add_scalar(tag=mode + 'rewards/stack', scalar_value=reward_vector[13].mean())

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
        if std.shape[0] == 1:
            self.add_scalar(tag='Std/touch', scalar_value=std.mean())
        else:
            # self.add_scalar(tag='Std/close', scalar_value=std[0].mean())
            # self.add_scalar(tag='Std/above', scalar_value=std[1].mean())
            # self.add_scalar(tag='Std/above_close', scalar_value=std[2].mean())
            # self.add_scalar(tag='Std/left', scalar_value=std[3].mean())
            # self.add_scalar(tag='Std/left_close', scalar_value=std[4].mean())
            # self.add_scalar(tag='Std/below', scalar_value=std[5].mean())
            # self.add_scalar(tag='Std/right', scalar_value=std[6].mean())
            # self.add_scalar(tag='Std/below_close', scalar_value=std[7].mean())
            # self.add_scalar(tag='Std/right_close', scalar_value=std[8].mean())
            # self.add_scalar(tag='Std/move1', scalar_value=std[9].mean())
            # self.add_scalar(tag='Std/move2', scalar_value=std[10].mean())
            # self.add_scalar(tag='Std/touch', scalar_value=std[11].mean())
            # self.add_scalar(tag='Std/no_touch', scalar_value=std[12].mean())
            # self.add_scalar(tag='Std/stack', scalar_value=std[13].mean())

            self.add_scalar(tag='Std/close', scalar_value=std[0].mean())
            self.add_scalar(tag='Std/above', scalar_value=std[1].mean())
            self.add_scalar(tag='Std/above_close', scalar_value=std[2].mean())
            self.add_scalar(tag='Std/left', scalar_value=std[3].mean())
            self.add_scalar(tag='Std/left_close', scalar_value=std[4].mean())
            self.add_scalar(tag='Std/move1', scalar_value=std[5].mean())
            self.add_scalar(tag='Std/move2', scalar_value=std[6].mean())
            self.add_scalar(tag='Std/touch', scalar_value=std[7].mean())
            self.add_scalar(tag='Std/no_touch', scalar_value=std[8].mean())
            self.add_scalar(tag='Std/stack', scalar_value=std[9].mean())

    def log_schedule_decisions(self, schedule_decisions):
        for i, d in enumerate(schedule_decisions):
            self.add_scalar(tag='Scheduler/decision' + str(i), scalar_value=d)
            
    def log_observations(self, observations):
        self.add_scalar(tag='observations/tcp_xpos', scalar_value=torch.max(observations[:, 0]))
        self.add_scalar(tag='observations/tcp_ypos', scalar_value=torch.max(observations[:, 1]))
        self.add_scalar(tag='observations/tcp_zpos', scalar_value=torch.max(observations[:, 2]))
        self.add_scalar(tag='observations/tcp_xvel', scalar_value=torch.max(observations[:, 3]))
        self.add_scalar(tag='observations/tcp_yvel', scalar_value=torch.max(observations[:, 4]))
        self.add_scalar(tag='observations/tcp_zvel', scalar_value=torch.max(observations[:, 5]))
        self.add_scalar(tag='observations/finger_pos1', scalar_value=torch.max(observations[:, 6]))
        self.add_scalar(tag='observations/finger_pos2', scalar_value=torch.max(observations[:, 7]))
        self.add_scalar(tag='observations/cubic_xpos', scalar_value=torch.max(observations[:, 8]))
        self.add_scalar(tag='observations/cubic_ypos', scalar_value=torch.max(observations[:, 9]))
        self.add_scalar(tag='observations/cubic_zpos', scalar_value=torch.max(observations[:, 10]))
        self.add_scalar(tag='observations/cuboid_xpos', scalar_value=torch.max(observations[:, 11]))
        self.add_scalar(tag='observations/cuboid_ypos', scalar_value=torch.max(observations[:, 12]))
        self.add_scalar(tag='observations/cuboid_zpos', scalar_value=torch.max(observations[:, 13]))
        self.add_scalar(tag='observations/cubic_xvel', scalar_value=torch.max(observations[:, 14]))
        self.add_scalar(tag='observations/cubic_yvel', scalar_value=torch.max(observations[:, 15]))
        self.add_scalar(tag='observations/cubic_zvel', scalar_value=torch.max(observations[:, 16]))
        self.add_scalar(tag='observations/cuboid_xvel', scalar_value=torch.max(observations[:, 17]))
        self.add_scalar(tag='observations/cuboid_yvel', scalar_value=torch.max(observations[:, 18]))
        self.add_scalar(tag='observations/cuboid_zvel', scalar_value=torch.max(observations[:, 19]))
        self.add_scalar(tag='observations/cubic_xrel_pos', scalar_value=torch.max(observations[:, 20]))
        self.add_scalar(tag='observations/cubic_xrel_pos', scalar_value=torch.max(observations[:, 21]))
        self.add_scalar(tag='observations/cubic_zrel_pos', scalar_value=torch.max(observations[:, 22]))
        self.add_scalar(tag='observations/cuboid_xrel_pos', scalar_value=torch.max(observations[:, 23]))
        self.add_scalar(tag='observations/cuboid_yrel_pos', scalar_value=torch.max(observations[:, 24]))
        self.add_scalar(tag='observations/cuboid_zrel_pos', scalar_value=torch.max(observations[:, 25]))
