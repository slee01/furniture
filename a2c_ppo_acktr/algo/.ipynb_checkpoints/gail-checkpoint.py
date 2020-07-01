import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd
import torch.distributions.normal as nrm

from baselines.common.running_mean_std import RunningMeanStd

from a2c_ppo_acktr import utils


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_latent, latent_dim, d_lr, device, good_end, extract_obs, action_space):
        super(Discriminator, self).__init__()

        self.device = device
        self.use_latent = use_latent
        self.latent_dim = latent_dim
        self.d_lr = d_lr
        self.good_end = good_end
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

        if use_latent:
            input_dim += latent_dim

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.trunk.train()
        self.optimizer = torch.optim.Adam(self.trunk.parameters(), lr=self.d_lr)

        if self.action_space == "Discrete":
            self.noise_dist = nrm.Normal(loc=0.0, scale=0.01)
        else:
            self.noise_dist = nrm.Normal(loc=0.0, scale=0.1)

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def forward(self, input):
        raise NotImplementedError("forward func() in standard Discriminator is not available yet")

    def compute_grad_pen(self,
                         expert_state,
                         expert_task,
                         expert_action,
                         policy_state,
                         policy_task,
                         policy_action,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_task, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_task, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)

        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()

        return grad_pen

    def update(self, posterior, expert_loader, rollouts, obsfilt=None):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        gail_losses = 0
        grad_losses = 0
        n = 0

        for expert_batch, policy_batch in zip(expert_loader, policy_data_generator):
            if self.use_latent:
                policy_states, policy_tasks, policy_actions = policy_batch[0], policy_batch[1], policy_batch[4].float()
            else:
                policy_states, policy_actions = policy_batch[0], policy_batch[2].float()

            expert_states, expert_actions = expert_batch

            if self.action_space == "Discrete":
                policy_actions = utils.get_one_hot_vector(policy_actions, self.action_dim)
                expert_actions = utils.get_one_hot_vector(expert_actions, self.action_dim)

            policy_states += self.noise_dist.sample(policy_states.shape).to(self.device)
            # policy_tasks, policy_task_features = posterior.act(policy_states, policy_actions, mean_mode=False)
            policy_inputs = torch.cat([policy_states, policy_tasks, policy_actions], dim=1)
            policy_ds = self.trunk(policy_inputs)

            half_point = int(expert_states.shape[-1] / 2)
            if self.extract_obs:
                expert_states = expert_states[:, :half_point]

            expert_states = obsfilt(expert_states.numpy(), update=True)
            expert_states = torch.FloatTensor(expert_states).to(self.device)
            expert_states += self.noise_dist.sample(expert_states.shape).to(self.device)

            expert_actions = expert_actions.to(self.device)
            expert_tasks, expert_task_features = posterior.act(expert_states, expert_actions, mean_mode=True)
            expert_tasks = expert_tasks.detach()

            expert_inputs = torch.cat([expert_states, expert_tasks, expert_actions], dim=1)
            expert_ds = self.trunk(expert_inputs)

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_ds, torch.ones(expert_ds.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_ds, torch.zeros(policy_ds.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_states, expert_tasks, expert_actions,
                                             policy_states, policy_tasks, policy_actions)

            loss += (gail_loss + grad_pen).item()
            n += 1

            gail_losses += gail_loss.item()
            grad_losses += grad_pen.item()

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()

        return loss / n, gail_losses / n, grad_losses / n

    def predict_reward(self, state, task, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()

            if self.action_space == "Discrete":
                action = utils.get_one_hot_vector(action, self.action_dim)

            d = self.trunk(torch.cat([state, task, action.float()], dim=1))
            s = torch.sigmoid(d)

            if self.good_end:
                # 0 for expert-like states, goes to -inf for non-expert-like states
                # compatible with envs with traj cutoffs for good (expert-like) behavior
                # e.g. mountain car, which gets cut off when the car reaches the destination
                reward = s.log()  # (-inf, 0) for envs where end option is GOOD transition

            else:
                # 0 for non-expert-like states, goes to +inf for expert-like states
                # compatible with envs with traj cutoffs for bad (non-expert-like) behavior
                # e.g. walking simulations that get cut off when the robot falls over
                if self.action_space == "Discrete":
                    reward = s.log() - (1 - s).log()  # (-inf, inf) for envs where end option is BAD transition(ikostrikov)
                else:
                    reward = - (1 - s).log()  # (0, inf)    for envs where end option is BAD transition

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward


class WassersteinDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_latent, latent_dim, d_lr, device, good_end, extract_obs, action_space):
        super(WassersteinDiscriminator, self).__init__()

        self.device = device
        self.use_latent = use_latent
        self.latent_dim = latent_dim
        self.d_lr = d_lr
        self.good_end = good_end
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

        if use_latent:
            input_dim += latent_dim

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.trunk.train()

        self.optimizer = torch.optim.RMSprop(self.trunk.parameters(), lr=self.d_lr)

        if self.action_space == "Discrete":
            self.noise_dist = nrm.Normal(loc=0.0, scale=0.01)
        else:
            self.noise_dist = nrm.Normal(loc=0.0, scale=0.1)

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def forward(self, input):
        raise NotImplementedError("forward func() in WassersteinD is not available yet")

    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)

        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()

        return grad_pen

    def update(self, posterior, expert_loader, rollouts, obsfilt=None):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(
            None, mini_batch_size=expert_loader.batch_size)

        loss = 0
        gail_losses = 0.0
        grad_losses = 0.0
        n = 0

        for expert_batch, policy_batch in zip(expert_loader, policy_data_generator):
            if self.use_latent:
                policy_states, policy_tasks, policy_actions = policy_batch[0], policy_batch[1], policy_batch[4].float()
            else:
                policy_states, policy_actions = policy_batch[0], policy_batch[2].float()

            expert_states, expert_actions = expert_batch

            if self.action_space == "Discrete":
                policy_actions = utils.get_one_hot_vector(policy_actions, self.action_dim)
                expert_actions = utils.get_one_hot_vector(expert_actions, self.action_dim)

            policy_states += self.noise_dist.sample(policy_states.shape).to(self.device)
            policy_inputs = torch.cat([policy_states, policy_actions], dim=1)
            policy_ds = self.trunk(policy_inputs)

            half_point = int(expert_states.shape[-1] / 2)
            if self.extract_obs:
                expert_states = expert_states[:, :half_point]

            expert_states = obsfilt(expert_states.numpy(), update=True)
            expert_states = torch.FloatTensor(expert_states).to(self.device)
            expert_states += self.noise_dist.sample(expert_states.shape).to(self.device)

            expert_actions = expert_actions.to(self.device)
            expert_tasks, expert_task_features = posterior.act(expert_states, expert_actions, mean_mode=True)

            expert_inputs = torch.cat([expert_states, expert_tasks, expert_actions], dim=1)
            expert_ds = self.trunk(expert_inputs)

            gail_loss = -torch.mean(policy_ds - expert_ds)
            grad_pen = self.compute_grad_pen(expert_states, expert_actions,
                                             policy_states, policy_actions)

            loss += (gail_loss + grad_pen).item()
            n += 1

            gail_losses += gail_loss.item()
            grad_losses += grad_pen.item()

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()

        return loss / n, gail_losses / n, grad_losses / n

    def predict_reward(self, state, task, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()

            if self.action_space == "Discrete":
                action = utils.get_one_hot_vector(action, self.action_dim)

            d = self.trunk(torch.cat([state, task, action.float()], dim=1))
            s = torch.sigmoid(d)

            if self.good_end:
                # 0 for expert-like states, goes to -inf for non-expert-like states
                # compatible with envs with traj cutoffs for good (expert-like) behavior
                # e.g. mountain car, which gets cut off when the car reaches the destination
                reward = s.log()  # (-inf, 0) for envs where end option is GOOD transition

            else:
                # 0 for non-expert-like states, goes to +inf for expert-like states
                # compatible with envs with traj cutoffs for bad (non-expert-like) behavior
                # e.g. walking simulations that get cut off when the robot falls over
                if self.action_space == "Discrete":
                    reward = s.log() - (1 - s).log()  # (-inf, inf) for envs where end option is BAD transition(ikostrikov)
                else:
                    reward = - (1 - s).log()  # (0, inf)    for envs where end option is BAD transition

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return
