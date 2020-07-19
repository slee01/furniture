import os
import time
from collections import deque
import numpy as np
import random
import copy

import torch

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.algo import posterior
from a2c_ppo_acktr.algo import task_trans_model
from a2c_ppo_acktr.algo import expert_dataset
from a2c_ppo_acktr.algo import policy_dataset
from a2c_ppo_acktr.algo import task_dataset
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy, HierarchicalPolicy
from a2c_ppo_acktr.storage import RolloutStorage, LatentRolloutStorage
from evaluation import evaluate, evaluate_with_latent

np.set_printoptions(precision=4)
np.set_printoptions(threshold=10)


def expand_save_folder_dirs(rootdir, args):
    latent_seed_pretrain = str(args.latent_dim) + "_" + str(args.seed) + "_" + args.pretrain_algo

    _filename = os.path.expanduser(rootdir)
    filename = os.path.join(_filename, args.algo, args.gail_algo, args.save_date, latent_seed_pretrain)

    try:
        os.makedirs(filename)
    except OSError:
        pass

    utils.cleanup_log_dir(filename)

    return filename


def expand_load_folder_dirs(rootdir, args):
    latent_seed_pretrain = str(args.latent_dim) + "_" + str(args.seed) + "_" + args.pretrain_algo

    _filename = os.path.expanduser(rootdir)
    filename = os.path.join(_filename, args.algo, args.gail_algo, args.load_date, latent_seed_pretrain)

    return filename


def reset_model_machines(model, device):
    model.to(device)
    model.device = device


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = expand_save_folder_dirs(args.log_dir, args)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(eval_log_dir)

    save_dir = expand_save_folder_dirs(args.save_dir, args)
    pretrain_dir = expand_load_folder_dirs(args.pretrain_dir, args)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args, args.env_name, args.seed, args.num_process, args.gamma, log_dir, device, False)

    obs_dim = int(envs.observation_space.shape[0])

    if len(envs.action_space.shape) > 0:
        act_dim = int(envs.action_space.shape[0])
    else:
        act_dim = envs.action_space.n

    print("************************************************************************************************")
    print("0. task: ", args.env_name, " device: ", device)
    print("   observation shape: ", envs.observation_space.shape)
    print("1. policy_lr & task_lr {}, b_lr: {}, discr_lr: {}, postr_lr: {}".format(args.lr, args.b_lr, args.d_lr,
                                                                                   args.p_lr))
    if args.load_model:  # load trained model and obs_rms
        if args.hierarchical_policy and args.pretrain_algo == 'cvae':
            _agent, ob_rms, post, discr, trans = \
                torch.load(os.path.join(pretrain_dir, args.env_name + ".pt"), map_location=device)

            print("Load Trained Model from ", os.path.join(pretrain_dir, args.env_name + ".pt"))

            agent = algo.HierarchicalPPO(
                _agent.actor_critic,
                device,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.bc_loss_coef,
                args.entropy_coef,
                args.extract_obs,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm
            )

        else:
            _agent, ob_rms = \
                torch.load(os.path.join(pretrain_dir, args.env_name + ".pt"), map_location=device)

            print("Load Trained Model from ", os.path.join(pretrain_dir, args.env_name + ".pt"))

            agent = algo.PPO(
                _agent.actor_critic,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm)

        if args.gail:
            assert len(envs.observation_space.shape) == 1
            if args.gail_algo == 'wasserstein':
                print("5. discriminator: wasserstein_discriminator")
                discr = gail.WassersteinDiscriminator(
                    input_dim=obs_dim + act_dim,
                    hidden_dim=100,
                    use_latent=args.use_latent,
                    latent_dim=args.latent_dim,
                    d_lr=args.d_lr,
                    device=device,
                    good_end=args.good_end,
                    extract_obs=args.extract_obs,
                    action_space=envs.action_space)
            elif args.gail_algo == 'standard':
                print("5. discriminator: standard_discriminator")
                discr = gail.Discriminator(
                    input_dim=obs_dim + act_dim,
                    hidden_dim=100,
                    use_latent=args.use_latent,
                    latent_dim=args.latent_dim,
                    d_lr=args.d_lr,
                    device=device,
                    good_end=args.good_end,
                    extract_obs=args.extract_obs,
                    action_space=envs.action_space)

        if args.posterior and args.reset_posterior:
            print("7. reset the posterior network: true")
            post = posterior.Posterior(
                input_dim=obs_dim + act_dim,
                hidden_dim=100,
                feature_dim=32,
                latent_dim=args.latent_dim,
                latent_space=args.latent_space,
                p_lr=args.p_lr,
                action_space=envs.action_space,
                device=device)

        if args.task_transition and args.reset_transition:
            print("7. reset the task transition model: true")
            trans = task_trans_model.TaskTransitionModel(input_dim=obs_dim + args.latent_dim,
                                                         hidden_dim=100,
                                                         feature_dim=32,
                                                         latent_dim=args.latent_dim,
                                                         latent_space=args.latent_space,
                                                         init_beta=args.init_beta,
                                                         t_lr=args.t_lr,
                                                         b_lr=args.b_lr,
                                                         i_c=args.i_c,
                                                         device=device,
                                                         extract_obs=args.extract_obs,
                                                         action_space=envs.action_space,
                                                         is_recurrent=True)

        reset_model_machines(agent.actor_critic, device)
        reset_model_machines(post, device)
        reset_model_machines(discr, device)
        reset_model_machines(trans, device)

        trans.i_c = args.i_c
        actor_critic = agent.actor_critic
        envs.ob_rms = ob_rms
        print("   information_botlleneck constraint: ", trans.i_c)
        print("************************************************************************************************")

    else:
        if args.hierarchical_policy:
            print("2. policy: hierarchical_policy")
            actor_critic = HierarchicalPolicy(
                envs.observation_space.shape,
                args.latent_dim,
                args.latent_space,
                envs.action_space,
                base_kwargs={'recurrent': args.recurrent_policy, 'latent_size': args.latent_dim})
        else:
            print("2. policy: standard_policy")
            actor_critic = Policy(
                envs.observation_space.shape,
                envs.action_space,
                base_kwargs={'recurrent': args.recurrent_policy})

        actor_critic.to(device)

        if args.algo == 'a2c':
            print("3. algorithm: a2c")
            agent = algo.A2C_ACKTR(
                actor_critic,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                alpha=args.alpha,
                max_grad_norm=args.max_grad_norm)
        elif args.algo == 'ppo':
            print("3. algorithm: ppo")
            if args.hierarchical_policy:
                agent = algo.HierarchicalPPO(
                    actor_critic,
                    device,
                    args.clip_param,
                    args.ppo_epoch,
                    args.num_mini_batch,
                    args.value_loss_coef,
                    args.bc_loss_coef,
                    args.entropy_coef,
                    args.extract_obs,
                    lr=args.lr,
                    eps=args.eps,
                    max_grad_norm=args.max_grad_norm)
            else:
                agent = algo.PPO(
                    actor_critic,
                    args.clip_param,
                    args.ppo_epoch,
                    args.num_mini_batch,
                    args.value_loss_coef,
                    args.entropy_coef,
                    lr=args.lr,
                    eps=args.eps,
                    max_grad_norm=args.max_grad_norm)
        elif args.algo == 'acktr':
            print("3. algorithm: acktr")
            agent = algo.A2C_ACKTR(
                actor_critic,
                args.value_loss_coef,
                args.entropy_coef,
                acktr=True)

        if args.posterior:
            print("4. posterior: true")
            post = posterior.Posterior(
                input_dim=obs_dim + act_dim,
                hidden_dim=100,
                feature_dim=32,
                latent_dim=args.latent_dim,
                latent_space=args.latent_space,
                p_lr=args.p_lr,
                action_space=envs.action_space,
                device=device)

        if args.gail:
            assert len(envs.observation_space.shape) == 1
            if args.gail_algo == 'wasserstein':
                print("5. discriminator: wasserstein_discriminator")
                discr = gail.WassersteinDiscriminator(
                    input_dim=obs_dim + act_dim,
                    hidden_dim=100,
                    use_latent=args.use_latent,
                    latent_dim=args.latent_dim,
                    d_lr=args.d_lr,
                    device=device,
                    good_end=args.good_end,
                    extract_obs=args.extract_obs,
                    action_space=envs.action_space)
            elif args.gail_algo == 'standard':
                print("5. discriminator: standard_discriminator")
                discr = gail.Discriminator(
                    input_dim=obs_dim + act_dim,
                    hidden_dim=100,
                    use_latent=args.use_latent,
                    latent_dim=args.latent_dim,
                    d_lr=args.d_lr,
                    device=device,
                    good_end=args.good_end,
                    extract_obs=args.extract_obs,
                    action_space=envs.action_space)

        if args.task_transition:
            print("6. task_transition_model: true")
            trans = task_trans_model.TaskTransitionModel(input_dim=obs_dim + args.latent_dim,
                                                         hidden_dim=100,
                                                         feature_dim=32,
                                                         latent_dim=args.latent_dim,
                                                         latent_space=args.latent_space,
                                                         init_beta=args.init_beta,
                                                         t_lr=args.t_lr,
                                                         b_lr=args.b_lr,
                                                         i_c=args.i_c,
                                                         device=device,
                                                         extract_obs=args.extract_obs,
                                                         action_space=envs.action_space,
                                                         is_recurrent=True)

    ##########################################################################################################
    # DEFINE DATA LOADER

    file_name = os.path.join(
        args.gail_experts_dir,
        "trajs_{}_{}.pt".format(args.env_name.split('-')[0].lower(), args.expert_algo))

    gail_train_loader = torch.utils.data.DataLoader(
        expert_dataset.ExpertDataset(args.env_name, file_name),
        batch_size=args.gail_batch_size,
        shuffle=True,
        drop_last=True)

    policy_train_loader = torch.utils.data.DataLoader(
        policy_dataset.PolicyDataset(args.env_name, file_name),
        batch_size=args.gail_batch_size,
        shuffle=True,
        drop_last=True)

    print("************************************************************************************************")
    if args.latent_space == 'continuous':
        prev_task = torch.normal(mean=torch.zeros(1, args.latent_dim), std=torch.ones(1, args.latent_dim)).to(device)
    elif args.latent_space == 'discrete':
        prev_task = torch.Tensor(random.sample(range(args.latent_dim), 1)).to(device)

    if args.hierarchical_policy:
        rollouts = LatentRolloutStorage(args.num_steps, args.num_processes,
                                        envs.observation_space.shape, envs.action_space, args.latent_dim,
                                        args.latent_space, actor_critic.recurrent_hidden_state_size, trans.hidden_dim)
    else:
        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                  envs.observation_space.shape, envs.action_space,
                                  actor_critic.recurrent_hidden_state_size)

    # DEFINE DATA LOADER
    ##########################################################################################################

    obs = envs.reset().to(device)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    epochs = []
    eval_epochs = []
    max_rewards, min_rewards, median_rewards, mean_rewards = [], [], [], []
    eval_max_rewards, eval_min_rewards, eval_median_rewards, eval_mean_rewards = [], [], [], []
    best_mean_rewards = -1000.0
    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(num_updates):
        epochs.append(j)

        if args.use_linear_lr_decay:
            utils.update_linear_schedule(agent.optimizer, j, num_updates,
                                         agent.optimizer.lr if args.algo == "acktr" else args.lr)

        
