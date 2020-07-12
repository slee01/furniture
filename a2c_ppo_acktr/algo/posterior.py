import torch
import torch.nn as nn
import torch.distributions.normal as nrm
import torch.utils.data

from baselines.common.running_mean_std import RunningMeanStd

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init

from a2c_ppo_acktr import utils


class Posterior(nn.Module):
    def __init__(self, input_dim, hidden_dim, feature_dim, latent_dim, latent_space, p_lr, action_space, device):
        super(Posterior, self).__init__()
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.latent_space = latent_space
        self.p_lr = p_lr
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, feature_dim)).to(device)

        if self.latent_space == 'discrete':
            self.dist = Categorical(self.feature_dim, self.latent_dim).to(self.device)
            print("   latent distribution is Categorical")
        elif self.latent_space == 'continuous':
            self.dist = DiagGaussian(self.feature_dim, self.latent_dim).to(self.device)
            print("   latent distribution is DiagGaussian")
        else:
            raise NotImplementedError

        self.trunk.train()
        self.optimizer = torch.optim.Adam(self.trunk.parameters(), lr=self.p_lr)

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def forward(self, input):
        raise NotImplementedError("forward func() in Posterior is not available yet")

    def split_parameters(self, output):
        raise NotImplementedError("split_parameters function in posterior is deprecated.")
        # assert output.shape[-1] % 2 == 0, "parameters are not an even number"
        #
        # halfpoint = output.shape[-1] // 2
        # mus, sigmas = output[:, :halfpoint], output[:, halfpoint:]
        # sigmas = 1e-6 + torch.sigmoid(sigmas)
        #
        # return mus, sigmas

    def evaluate(self, obs, action):
        raise NotImplementedError("evaluate function in posterior is deprecated.")
        # assert len(obs) == len(action), "len(obs) != len(action)"
        #
        # params = self.trunk(torch.cat([obs, action.float()], dim=1))
        # mus, sigmas = self.split_parameters(params)
        #
        # return mus, sigmas

    def act(self, states, actions, mean_mode=False):
        task_features = self.trunk(torch.cat([states, actions.float()], dim=1))
        dist = self.dist(task_features)
        if mean_mode:
            tasks = dist.mode().float()
        else:
            tasks = dist.sample().float()
    
        # return tasks, task_mus, task_sigmas, task_hxs
        return tasks, task_features

    def get_dist_params(self, latent_features):
        if self.latent_space == 'discrete':
            logits = self.dist.get_logits(latent_features)
            return logits
        elif self.latent_space == 'continuous':
            dist = self.dist(latent_features)
            mus, sigmas = dist.mean, dist.stddev
            return mus, sigmas
        else:
            raise NotImplementedError
        
    def posterior_loss(self, predict_latents, rollout_latents):
        if self.latent_space == 'discrete':
            predict_logits = self.get_dist_params(predict_latents)
            rollout_latents = torch.squeeze(rollout_latents.to(self.device, torch.long))
            loss = self.cross_entropy_loss(predict_logits, rollout_latents)
        elif self.latent_space == 'continuous':
            mus, sigmas = self.get_dist_params(predict_latents)
            log_likelihood = -torch.mean(0.5 * torch.log(sigmas.pow(2)) + ((rollout_latents - mus).pow(2)/(2.0*sigmas.pow(2))))
            loss = -log_likelihood

        return loss

    def update(self, rollouts, batch_size):
        self.train()

        policy_data_generator = rollouts.feed_forward_generator(None, mini_batch_size=batch_size)

        loss = 0
        n = 0

        for rollout_batch in policy_data_generator:

            rollout_state, rollout_latent, rollout_action = rollout_batch[0], rollout_batch[1], rollout_batch[4]

            if self.action_space == "Discrete":
                rollout_action = utils.get_one_hot_vector(rollout_action, self.action_dim)

            # policy_mus, policy_sigmas = self.evaluate(policy_state, policy_action.float())
            # posterior_loss = self.pos_nll_loss(policy_mus, policy_sigmas, policy_latent)
            latent_feature = self.trunk(torch.cat([rollout_state, rollout_action.float()], dim=1))
            posterior_loss = self.posterior_loss(latent_feature, rollout_latent)

            loss = loss + posterior_loss.item()
            n = n + 1

            self.optimizer.zero_grad()
            posterior_loss.backward()
            self.optimizer.step()

        return loss / n

    def predict_reward(self, state, action, latent, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()

            if self.action_space == "Discrete":
                action = utils.get_one_hot_vector(action, self.action_dim)

            latent_feature = self.trunk(torch.cat([state, action.float()], dim=1))
            # print("latent_feature: ", latent_feature, latent_feature.shape)
            dist = self.dist(latent_feature)
            # print("given_latent: ", latent, latent.shape)
            reward = dist.log_probs(latent)
            # print("reward: ", reward, reward.shape)
            reward = torch.mean(reward)
            # print("reward: ", reward, reward.shape)
            
            # if self.latent_space == 'discrete':
            #     reward = torch.mean(dist.log_probs(latent))
            # elif self.latent_space == 'continuous':
            #     reward = torch.mean(dist.log_probs(latent))
            # else:
            #     raise NotImplementedError
            
            # mus, sigmas = self.evaluate(state, action.float())
            # log_likelihood = -torch.mean(0.5 * torch.log(sigmas.pow(2)) + ((latent - mus).pow(2) / (2.0 * sigmas.pow(2))))
            # reward = log_likelihood

            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            # return (reward - self.ret_rms.mean[0]) / np.sqrt(self.ret_rms.var[0] + 1e-8)
            return reward

    def relabel_task(self, state, action, deterministic=False):
        with torch.no_grad():
            self.eval()

            if self.action_space == "Discrete":
                action = utils.get_one_hot_vector(action, self.action_dim)

            task_feature = self.trunk(torch.cat([state, action.float()], dim=1))
            dist = self.dist(task_feature)

            if deterministic:
                task = dist.mode()
            else:
                task = dist.sample()

        return task

