from a2c_ppo_acktr.model import TaskMLPBase
from a2c_ppo_acktr import utils

import torch
import torch.nn as nn
import torch.utils.data

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init

from baselines.common.running_mean_std import RunningMeanStd


class TaskTransitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, feature_dim, latent_dim, latent_space, init_beta, t_lr, b_lr, i_c,
                 device, extract_obs, action_space, is_recurrent=True):
        super(TaskTransitionModel, self).__init__()
        self.action_space = action_space.__class__.__name__
        self.latent_space = latent_space

        if self.action_space == "Discrete":
            self.action_dim = action_space.n
        elif self.action_space == "Box":
            self.action_dim = action_space.shape[0]
        elif self.action_space == "MultiBinary":
            self.action_dim = action_space.shape[0]
        else:
            raise NotImplementedError

        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.t_lr = t_lr
        self.b_lr = b_lr
        self.i_c = i_c
        self.beta = init_beta
        self.extract_obs = extract_obs
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim), nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh(),
            nn.Linear(self.hidden_dim, self.feature_dim)).to(device)

        # self.trunk = TaskMLPBase(input_size=input_dim,
        #                          recurrent=is_recurrent,
        #                          hidden_size=self.hidden_dim,
        #                          output_size=self.feature_dim).to(device)

        if self.latent_space == 'discrete':
            self.dist = Categorical(self.feature_dim, self.latent_dim).to(self.device)
            print("   latent distribution is Categorical")
        elif self.latent_space == 'continuous':
            self.dist = DiagGaussian(self.feature_dim, self.latent_dim).to(self.device)
            print("   latent distribution is DiagGaussian")
        else:
            raise NotImplementedError

        self.trunk.train()
        self.optimizer = torch.optim.Adam(self.trunk.parameters(), lr=self.t_lr)

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def forward(self, input):
        raise NotImplementedError("forward function in task transition model is not available yet")

    def split_parameters(self, output):
        raise NotImplementedError("split_parameters function in task transition model is deprecated.")
        # assert output.shape[-1] % 2 == 0, "parameters are not an even number"
        #
        # halfpoint = output.shape[-1] // 2
        # mus, log_vars = output[:, :halfpoint], output[:, halfpoint:]
        # sigmas = torch.exp(0.5 * log_vars)  # + 1e-6
        #
        # return mus, sigmas

    def sample_task(self, mus, sigmas, mean_mode=False):
        raise NotImplementedError("sample_task function in task transition model is deprecated.")
        # if not mean_mode:
        #     out = (torch.randn_like(mus) * sigmas) + mus
        # else:
        #     out = mus
        #
        # return out

    # def act(self, obs, prev_tasks, rnn_hxs, masks, mean_mode=False,
    def act(self, obs, prev_tasks, mean_mode=False,
            use_random_latent=False, use_constant_latent=False, constant_latent=1.0):

        if self.latent_space == "discrete":
            prev_tasks = utils.get_one_hot_vector(prev_tasks, self.latent_dim)

        inputs = torch.cat((obs, prev_tasks), dim=-1)
        # task_features, task_hxs = self.trunk(inputs, rnn_hxs, masks)
        task_features = self.trunk(inputs)

        # task_mus, task_sigmas = self.split_parameters(task_features)
        # tasks = self.sample_task(task_mus, task_sigmas, mean_mode)
        dist = self.dist(task_features)
        if mean_mode:
            tasks = dist.mode().float()
        else:
            tasks = dist.sample().float()

        if use_random_latent:
            tasks = torch.randn_like(tasks)
        elif use_constant_latent:
            tasks = torch.ones_like(tasks) * constant_latent

        # return tasks, task_mus, task_sigmas, task_hxs
        # return tasks, task_features, task_hxs
        return tasks, task_features

    def get_dist_params(self, task_features):
        if self.latent_space == 'discrete':
            logits = self.dist.get_logits(task_features)
            return logits
        elif self.latent_space == 'continuous':
            dist = self.dist(task_features)
            mus, sigmas = dist.mean, dist.stddev
            return mus, sigmas
        else:
            raise NotImplementedError

    def evaluate_tasks(self, obs, prev_tasks, tasks):
        raise NotImplementedError("evaluate_tasks function in task transition model is deprecated.")
        # task_features = self.trunk(torch.cat([obs, prev_tasks], dim=1))
        # task_mus, task_sigmas = self.split_parameters(task_features)
        # task_log_probs = -0.5 * torch.log(task_sigmas.pow(2)) + ((tasks - task_mus).pow(2)/(2.0*task_sigmas.pow(2)))
        #
        # return task_log_probs

    def task_loss(self, task_features, expert_tasks):
        if self.latent_space == 'discrete':
            task_logits = self.get_dist_params(task_features)
            expert_tasks = torch.squeeze(expert_tasks.to(self.device, torch.long))
            loss = self.cross_entropy_loss(task_logits, expert_tasks)
        elif self.latent_space == 'continuous':
            mus, sigmas = self.get_dist_params(task_features)
            log_likelihood = -torch.mean(0.5 * torch.log(sigmas.pow(2)) + ((expert_tasks - mus).pow(2) / (2.0 * sigmas.pow(2))))
            loss = -log_likelihood
        else:
            raise NotImplementedError

        return loss

    def bottleneck_loss(self, task_features, i_c, alpha=1e-8):
        if self.latent_space == 'discrete':
            dist = self.dist(task_features)
            task_probs = dist.probs
            uniform_probs = torch.ones_like(task_probs) * (1.0 / self.latent_dim)
            # print("task_probs: ",  task_probs, task_probs.shape)
            # print("uniform_probs: ", uniform_probs, uniform_probs.shape)
            bottleneck_loss = task_probs * (task_probs / uniform_probs).log()
            # print("bottleneck_loss: ", bottleneck_loss, bottleneck_loss.shape)
            bottleneck_loss = torch.sum(bottleneck_loss)
        elif self.latent_space == 'continuous':
            mus, sigmas = self.get_dist_params(task_features)
            kl_divergence = (0.5 * torch.sum((mus ** 2) + (sigmas ** 2) - torch.log((sigmas ** 2) + alpha) - 1, dim=1))
            bottleneck_loss = (torch.mean(kl_divergence) - i_c)
        else:
            raise NotImplementedError

        return bottleneck_loss

    def optimize_beta(self, bottleneck_loss):
        self.beta = max(0, self.beta + (self.b_lr * bottleneck_loss.detach()))

    def update(self, rollouts, agent, obsfilt=None):
        self.train()

        actor_critic = agent.actor_critic

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        if actor_critic.is_recurrent:
            data_generator = rollouts.recurrent_generator(advantages, agent.num_mini_batch)
        else:
            data_generator = rollouts.feed_forward_generator(advantages, agent.num_mini_batch)

        task_loss_epoch = 0
        ib_loss_epoch = 0
        n = 0

        for sample in data_generator:
            obs_batch, tasks_batch, prev_tasks_batch, \
                recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, \
                old_action_log_probs_batch, adv_targ = sample

            estimate_tasks_batch, estimate_tasks_feature = self.act(obs_batch, prev_tasks_batch)
            _, estimate_actions_batch, _, _ = actor_critic.act(obs_batch,
                                                               estimate_tasks_batch,
                                                               recurrent_hidden_states_batch,
                                                               masks_batch,
                                                               deterministic=False)

            # Reshape to do in a single forward pass for all steps
            # def evaluate_actions(self, inputs, latents, rnn_hxs, masks, action):
            values, _, _, _ = actor_critic.evaluate_actions(obs_batch,
                                                            estimate_tasks_batch,
                                                            recurrent_hidden_states_batch,
                                                            masks_batch,
                                                            estimate_actions_batch.detach())

            # Define Task Transition Model loss
            if agent.use_clipped_value_loss:
                value_pred_clipped = value_preds_batch + \
                                     (values - value_preds_batch).clamp(-agent.clip_param, agent.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                task_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            else:
                task_loss = 0.5 * (return_batch - values).pow(2).mean()

            # Define Information Bottleneck loss
            ib_loss = self.bottleneck_loss(estimate_tasks_feature, self.i_c)

            self.optimizer.zero_grad()
            # (task_loss + ib_loss * self.beta).backward()
            task_loss.backward()
            self.optimizer.step()

            self.optimize_beta(bottleneck_loss=ib_loss)

            task_loss_epoch += task_loss.item()
            ib_loss_epoch += ib_loss.item()
            n = n + 1

        task_loss_epoch /= n
        ib_loss_epoch /= n

        return task_loss_epoch, ib_loss_epoch, self.beta

    def predict_reward(self, state, prev_task, task, gamma, mask, mean_mode=True, update_rms=True):
        with torch.no_grad():
            self.eval()

            if self.latent_space == "discrete":
                prev_task = utils.get_one_hot_vector(prev_task, self.latent_dim)

            input = torch.cat((state, prev_task), dim=-1)
            # task_features, task_hxs = self.trunk(input, rnn_hxs, mask)
            task_features = self.trunk(input)

            dist = self.dist(task_features)

            if mean_mode:
                pred_task = dist.mode().float()
            else:
                pred_task = dist.sample().float()

            reward = 0.5 * (task - pred_task).pow(2)
            reward = torch.mean(torch.sum(reward, dim=1))

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * mask * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward
