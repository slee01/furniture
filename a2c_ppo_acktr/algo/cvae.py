# prerequisites
import copy
import glob
import sys
import os
import time
from collections import deque

import gym
import custom_tasks

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.algo import posterior
from a2c_ppo_acktr.model import Policy, HierarchicalPolicy
from a2c_ppo_acktr.algo import expert_dataset
from a2c_ppo_acktr import algo, utils

import torch.distributions.normal as nrm


class CVAE(nn.Module):
    def __init__(self, agent, posterior, extract_obs, device, action_space, lr=0.001):
        super(CVAE, self).__init__()

        self.actor_critic = agent.actor_critic
        self.posterior = posterior
        self.extract_obs = extract_obs
        self.action_space = action_space.__class__.__name__

        if self.action_space == "Discrete":
            self.action_dim = action_space.n
        elif self.action_space == "Box":
            self.action_dim = action_space.shape[0]
        elif self.action_space == "MultiBinary":
            self.action_dim = action_space.shape[0]
        else:
            raise NotImplementedError

        self.device = device
        self.lr = lr
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.noise_dist = nrm.Normal(loc=0.0, scale=0.01)

        self.optimizer = optim.Adam(
            params=list(self.actor_critic.parameters()) + list(self.posterior.parameters()),
            lr=lr)

    def sampling(self, mu, sigmas):
        eps = torch.randn_like(sigmas)
        return eps.mul(sigmas).add(mu)

    def forward(self, states, one_hot_actions):
        mus, sigmas = self.posterior.evaluate(states, one_hot_actions)
        sub_tasks = self.sampling(mus, sigmas)

        # decoder_input = torch.cat([states, sub_tasks])
        if self.action_space == "Discrete":
            actions, action_logits = self.actor_critic.get_action(states, sub_tasks, deterministic=True)

            return actions, action_logits, mus, sigmas
        else:
            actions, action_mus, action_sigmas = self.actor_critic.get_action(states, sub_tasks, deterministic=True)
            return actions, action_mus, action_sigmas, mus, sigmas

    def continuous_reconstruction(self, policy_mus, policy_sigmas, expert_actions):
        log_likelihood = \
            -torch.mean(0.5 * torch.log(policy_sigmas.pow(2)) + \
                        ((expert_actions - policy_mus).pow(2)/(2.0*policy_sigmas.pow(2))))
        loss = -log_likelihood

        return loss

    def discrete_reconstruction(self, policy_logits, expert_actions):
        loss = self.cross_entropy_loss(policy_logits, expert_actions)

        return loss

    def continuous_loss(self, policy_mus, policy_sigmas, expert_actions, task_mus, task_sigmas, alpha=1e-8):
        RCE = self.continuous_reconstruction(policy_mus, policy_sigmas, expert_actions)
        KLD = 0.5 * torch.sum((task_mus ** 2) + (task_sigmas ** 2) - torch.log((task_sigmas ** 2) + alpha) - 1, dim=1)
        KLD = torch.sum(KLD)
        # KLD = torch.mean(KLD)  # ONLY for OldCarContinuous(it's the scale issue)

        return RCE, KLD

    def discrete_loss(self, policy_logits, expert_actions, task_mus, task_sigmas, alpha=1e-8):
        RCE = self.discrete_reconstruction(policy_logits, torch.squeeze(expert_actions.to(self.device, torch.long)))
        KLD = 0.5 * torch.sum((task_mus ** 2) + (task_sigmas ** 2) - torch.log((task_sigmas ** 2) + alpha) - 1, dim=1)
        KLD = torch.sum(KLD)
        # KLD = 0.1 * torch.mean(KLD)  # ONLY for OldCarContinuous(it's the scale issue)

        return RCE, KLD

    def update(self, expert_loader, obsfilt=None):
        self.train()
        rce_loss_epoch = 0
        reg_loss_epoch = 0

        for batch_idx, expert_batch in enumerate(expert_loader):
            expert_state, expert_action = expert_batch

            half_point = int(expert_state.shape[-1] / 2)
            if self.extract_obs:
                expert_state = expert_state[:, :half_point]

            expert_state = obsfilt(expert_state.numpy(), update=True)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_state += self.noise_dist.sample(expert_state.shape).to(self.device)
            expert_action = torch.FloatTensor(expert_action).to(self.device)

            if self.action_space == "Discrete":
                one_hot_expert_action = utils.get_one_hot_vector(expert_action, self.action_dim)

                policy_actions, policy_logits, task_mus, task_sigmas = self.forward(expert_state, one_hot_expert_action)
                rce_loss, reg_loss = self.discrete_loss(policy_logits, expert_action, task_mus, task_sigmas)
            else:
                policy_actions, policy_mus, policy_sigmas, task_mus, task_sigmas = self.forward(expert_state, expert_action)
                rce_loss, reg_loss = self.continuous_loss(policy_mus, policy_sigmas, expert_action, task_mus, task_sigmas)

            self.optimizer.zero_grad()
            (rce_loss + reg_loss).backward()
            self.optimizer.step()

            rce_loss_epoch += rce_loss.item()
            reg_loss_epoch += reg_loss.item()

        rce_loss_epoch /= batch_idx
        reg_loss_epoch /= batch_idx

        return rce_loss_epoch, reg_loss_epoch
