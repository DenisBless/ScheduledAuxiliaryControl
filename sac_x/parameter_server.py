import torch
from typing import Union, List
from torch.multiprocessing import Condition
from functools import reduce

from sac_x.actor_critic_nets import Actor, Critic
from sac_x.optimizer import SharedAdam


class ParameterServer:
    """
    Shared parameter server. Let g be the gradient of the shared network, g' the incoming gradient of a worker and G
    the fixed number of gradients until a update to the shared network parameters p is performed. The procedure is
    as follows:

    repeat until convergence:

        while i < G do:
            g += g' / G
            i++

        p -= η * g
    """

    def __init__(self, parser_args, worker_cv: Condition, server_cv: Condition):

        self.num_actions = parser_args.num_actions
        self.num_obs = parser_args.num_observations
        self.num_intentions = parser_args.num_intentions
        self.G = parser_args.num_workers * parser_args.num_grads  # number of gradients before updating networks

        self.N = torch.tensor(0)  # current number of gradients
        self.N.share_memory_()
        self.worker_cv = worker_cv
        self.server_cv = server_cv

        self.shared_actor = Actor(parser_args=parser_args)#.to('cuda:0')
        self.shared_actor.share_memory()

        self.shared_critic = Critic(parser_args=parser_args)#.to('cuda:0')
        self.shared_critic.share_memory()

        self.actor_grads, self.critic_grads = self.init_grad()

        self.actor_optimizer = SharedAdam(self.shared_actor.parameters(), parser_args.actor_lr)
        self.actor_optimizer.share_memory()
        self.critic_optimizer = SharedAdam(self.shared_critic.parameters(), parser_args.critic_lr)
        self.critic_optimizer.share_memory()

        self.global_gradient_norm = parser_args.global_gradient_norm

    def run(self) -> None:
        print("Parameter server started.")
        while True:
            with self.server_cv:
                self.server_cv.wait_for(lambda: self.N == self.G)
                self.N.zero_()
                self.update_params()
                self.worker_cv.notify_all()

    def receive_gradients(self, actor_grads, critic_grads) -> None:
        """
        Receive gradients by the workers.

        Args:

        Returns:
            No return value
        """
        with self.worker_cv:
            self.add_gradients(actor_grads=actor_grads, critic_grads=critic_grads)
            self.N += 1

    def add_gradients(self, actor_grads, critic_grads) -> None:
        for shared_ag, ag in zip(self.actor_grads, actor_grads):
            shared_ag += ag.cpu() / self.G
        for shared_cg, cg in zip(self.critic_grads, critic_grads):
            shared_cg += cg.cpu() / self.G

    def update_params(self) -> None:
        """
        Update the parameter of the shared actor and critic networks.

        Returns:
            No return value
        """
        for a_param, a_grad in zip(self.shared_actor.parameters(), self.actor_grads):
            a_param.grad = a_grad
        for c_param, c_grad in zip(self.shared_critic.parameters(), self.critic_grads):
            c_param.grad = c_grad

        if self.global_gradient_norm != -1:
            torch.nn.utils.clip_grad_norm_(self.shared_actor.parameters(), self.global_gradient_norm)
            torch.nn.utils.clip_grad_norm_(self.shared_critic.parameters(), self.global_gradient_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.zero_grads()

        assert not self.shared_critic.grad_norm
        assert not self.shared_actor.grad_norm

    def init_grad(self) -> Union[List, List]:
        actor_grads = [torch.zeros_like(x, requires_grad=False) for x in list(self.shared_actor.parameters())]
        critic_grads = [torch.zeros_like(x, requires_grad=False) for x in list(self.shared_critic.parameters())]
        for a, c in zip(actor_grads, critic_grads):
            a.share_memory_()
            c.share_memory_()
        return [actor_grads, critic_grads]

    def zero_grads(self) -> None:
        for a, c in zip(self.actor_grads, self.critic_grads):
            a.zero_()
            c.zero_()

    def get_grad_norm(self) -> Union[float, float]:
        ag_norm = reduce(lambda x, y: torch.norm(x) + torch.norm(y), self.actor_grads).item()
        cg_norm = reduce(lambda x, y: torch.norm(x) + torch.norm(y), self.critic_grads).item()
        return [ag_norm, cg_norm]

    def get_param_norm(self) -> Union[float, float]:
        ap_norm = reduce(lambda x, y: torch.norm(x) + torch.norm(y), list(self.shared_actor.parameters())).item()
        cp_norm = reduce(lambda x, y: torch.norm(x) + torch.norm(y), list(self.shared_critic.parameters())).item()
        return [ap_norm, cp_norm]
