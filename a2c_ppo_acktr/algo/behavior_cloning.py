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
from a2c_ppo_acktr import utils

import torch.distributions.normal as nrm


class BehaviorCloning(nn.Module):
    def __init__(self, agent, extract_obs, device, action_space, lr=0.001):
        super(BehaviorCloning, self).__init__()

        self.actor_critic = agent.actor_critic
        self.extract_obs= extract_obs
        self.action_space = action_space.__class__.__name__
        self.action_dim = action_space.n
        self.device = device
        self.lr = lr
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.noise_dist = nrm.Normal(loc=0.0, scale=0.1)
        self.optimizer = optim.Adam(params=self.actor_critic.parameters(), lr=lr)

    def update(self, pre_train_loader, obsfilt=None):
        supervised_loss_epoch = 0
        n = 0

        for expert_batch in pre_train_loader:
            expert_state, expert_action = expert_batch

            # if self.action_space == "Discrete":
            #     expert_action = utils.get_one_hot_vector(expert_action, self.action_dim)

            half_point = int(expert_state.shape[-1] / 2)
            if self.extract_obs:
                expert_state = expert_state[:, :half_point]

            expert_state = obsfilt(expert_state.numpy(), update=True)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = torch.FloatTensor(expert_action).to(self.device)

            if self.action_space == "Discrete":
                expert_action = torch.squeeze(expert_action.to(self.device, torch.long))
                policy_action, policy_logits = self.actor_critic.get_action(expert_state, deterministic=True)
                cross_entropy = self.cross_entropy_loss(policy_logits, expert_action)

                supervised_loss = cross_entropy
            else:
                policy_action, mu, sigma = self.actor_critic.get_action(expert_state, deterministic=True)

                log_likelihood = -torch.mean(
                    0.5 * torch.log(sigma.pow(2)) + ((expert_action - mu).pow(2) / (2.0 * sigma.pow(2))))

                supervised_loss = -log_likelihood

            # print("expert_states: ", expert_state.shape)
            # # print(expert_state[0])
            # print("tasks: ", tasks.shape)
            # # print(tasks[0])
            # print("expert_actions: ", expert_action.shape)
            # # print(expert_action[0])
            # print("policy_action: ", policy_action.shape)
            # # print(policy_action[0])

            # Update each network parameters
            self.optimizer.zero_grad()
            supervised_loss.backward()
            # nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            supervised_loss_epoch += supervised_loss.item()
            n += 1

        supervised_loss_epoch /= n

        return supervised_loss_epoch
