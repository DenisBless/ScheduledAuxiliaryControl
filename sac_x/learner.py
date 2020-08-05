import copy
import torch
from torch.multiprocessing import current_process
from torch.utils.tensorboard import SummaryWriter

from sac_x.loss_fn import ActorLoss, Retrace
from sac_x.replay_buffer import SharedReplayBuffer
from sac_x.parameter_server import ParameterServer


class Learner:
    def __init__(self,
                 actor: torch.nn.Module,
                 critic: torch.nn.Module,
                 parameter_server: ParameterServer,
                 replay_buffer: SharedReplayBuffer,
                 parser_args,
                 logger: SummaryWriter = None):

        self.actor = actor
        self.critic = critic

        self.target_actor = copy.deepcopy(self.actor)
        self.target_actor.freeze_net()
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic.freeze_net()

        self.parameter_server = parameter_server
        self.replay_buffer = replay_buffer
        self.cv = self.parameter_server.worker_cv

        self.num_actions = parser_args.num_actions
        self.num_obs = parser_args.num_observations

        self.logger = logger
        self.log_every = 10

        self.actor_loss = ActorLoss(alpha=parser_args.entropy_reg, num_intentions=parser_args.num_intentions)
        self.critic_loss = Retrace(num_actions=self.num_actions, num_intentions=parser_args.num_intentions)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), parser_args.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), parser_args.critic_lr)

        self.update_targnets_every = parser_args.update_targnets_every
        self.learning_steps = parser_args.learning_steps
        self.global_gradient_norm = parser_args.global_gradient_norm
        self.num_grads = parser_args.num_grads
        self.grad_ctr = 0

        if parser_args.num_workers > 1:
            self.process_id = current_process()._identity[0]  # process ID
        else:
            self.process_id = 1

    def run(self) -> None:
        """
        Calculates gradients w.r.t. the actor and the critic and sends them to a shared parameter server. Whenever
        the server has accumulated G gradients, the parameter of the shared critic and actor are updated and sent
        to the worker. However, the parameters of the shared actor and critic are copied to the worker after each
        iteration since it is unknown to the worker when the gradient updates were happening.

        Returns:
            No return value
        """

        self.actor.copy_params(self.parameter_server.shared_actor)
        self.critic.copy_params(self.parameter_server.shared_critic)

        self.actor.train()
        self.critic.train()

        for i in range(self.learning_steps):

            # Update the target networks
            if i % self.update_targnets_every == 0:
                self.update_targnets()

            states, actions, rewards, behaviour_log_pr, schedule_decisions = self.replay_buffer.sample()

            # Q(a_t, s_t)
            batch_Q = self.critic(actions, states)

            # Q_target(a_t, s_t)
            target_Q = self.target_critic(actions, states)

            # Compute 𝔼_π_target [Q(s_t,•)] with a ~ π_target(•|s_t), log(π_target(a|s)) with 1 sample
            mean, log_std = self.target_actor(states)

            action_sample, _ = self.target_actor.action_sample(mean, log_std)
            expected_target_Q = self.target_critic(action_sample, states)

            # log(π_target(a_t | s_t))
            target_action_log_prob = self.target_actor.get_log_prob(actions=actions, mean=mean, log_std=log_std)

            # a ~ π(•|s_t), log(π(a|s_t))
            current_mean, current_log_std = self.actor(states)
            current_actions, current_action_log_prob = self.actor.action_sample(current_mean, current_log_std)

            # Q(a, s_t)
            current_Q = self.critic(current_actions, states)

            critic_loss = self.critic_loss(Q=batch_Q,
                                           expected_target_Q=expected_target_Q,
                                           target_Q=target_Q,
                                           rewards=rewards,
                                           target_policy_probs=target_action_log_prob,
                                           behaviour_policy_probs=behaviour_log_pr,
                                           logger=self.logger)
            actor_loss = self.actor_loss(Q=current_Q, action_log_prob=current_action_log_prob.unsqueeze(-1))

            # Calculate gradients
            critic_grads = torch.autograd.grad(critic_loss, list(self.critic.parameters()), retain_graph=True)
            actor_grads = torch.autograd.grad(actor_loss, list(self.actor.parameters()), retain_graph=True)

            self.parameter_server.receive_gradients(actor_grads, critic_grads)

            self.grad_ctr += 1

            if self.grad_ctr == self.num_grads:
                with self.cv:
                    if self.parameter_server.N == self.parameter_server.G:
                        self.parameter_server.server_cv.notify()

                    self.cv.wait_for(lambda: self.parameter_server.N.item() == 0)

                    self.actor.copy_params(self.parameter_server.shared_actor)
                    self.critic.copy_params(self.parameter_server.shared_critic)

                    self.grad_ctr = 0

            # Keep track of different values
            if (self.process_id == 1) and (self.logger is not None) and (i % self.log_every == 0):
                self.logger.add_scalar(tag='Loss/Critic', scalar_value=critic_loss)
                self.logger.add_scalar(tag='Loss/Actor', scalar_value=actor_loss)
                self.logger.log_rewards(rewards)

        self.actor.copy_params(self.parameter_server.shared_actor)
        self.critic.copy_params(self.parameter_server.shared_critic)

    def update_targnets(self) -> None:
        """
        Update the target actor and the target critic by copying the parameter from the updated networks. I
        """
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

