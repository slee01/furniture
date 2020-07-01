import copy
import glob
import sys
import os
import time
from collections import deque

import gym
import custom_tasks

import numpy as np
import torch

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import cvae
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.algo import posterior
from a2c_ppo_acktr.algo import behavior_cloning
from a2c_ppo_acktr.algo import task_trans_model
from a2c_ppo_acktr.algo import expert_dataset
from a2c_ppo_acktr.algo import task_dataset
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy, HierarchicalPolicy
from evaluation import evaluate, evaluate_with_latent

# These options determine the way floating point numbers, arrays and other NumPy objects are displayed.
np.set_printoptions(precision=4)
# np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(threshold=10)

"""
act
x:  torch.Size([4, 2])     (num_processes, dim)
hxs:  torch.Size([4, 64])  (num_processes, dim)
masks:  torch.Size([4, 1]) (num_processes, dim)

evaluate_actions
x:  torch.Size([512, 2])      (num_steps * per_envs_num, dim)
hxs:  torch.Size([1, 64])     (1 * per_envs_num, dim) // hxs of first step in num_steps
masks:  torch.Size([512, 1])  (num_steps * per_envs_num, dim)
per_envs_num = num_processes / num_mini_batch
"""

"""
<Data>
Download from https://drive.google.com/open?id=1Ipu5k99nwewVDG1yFetUxqtwVlgBg5su
and store in this folder.

<RUN Pre-Training Actor & Posterior & Task Transition Model via CVAE>
python pretrain.py --env-name "FetchPickAndPlace-v1" --log-interval 1 \
--lr 3e-4 --use-linear-lr-decay --use-proper-time-limits \
--gail --expert-algo her --save-date 191025_CVAE \
--task-transition --use-latent  --hierarchical-policy --latent_dim 1 --posterior

python pretrain.py --env-name "FetchPickAndPlace-v1" --log-interval 1 \
--lr 3e-4 --use-linear-lr-decay --use-proper-time-limits \
--pretrain-algo bc \
--gail --expert-algo her --save-date 191029 \
--task-transition --use-latent  --hierarchical-policy --latent_dim 1 --posterior

python pretrain.py --env-name "FetchPickAndPlace-v1" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 \
--lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 \
--num-env-steps 10000000 --use-linear-lr-decay --use-proper-time-limits \
--gail --gail-algo standard --expert-algo her --save-date 191029 \
--pretrain-algo cvae \
--task-transition --use-latent  --hierarchical-policy --latent_dim 1 --posterior --num-process 1 --eval-interval 1

"""


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


def pretrain():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Define directories
    log_dir = expand_save_folder_dirs(args.log_dir, args)
    pre_log_dir = expand_save_folder_dirs(args.pre_log_dir, args)
    eval_log_dir = log_dir + "_eval"
    pre_eval_log_dir = pre_log_dir + "_eval"
    utils.cleanup_log_dir(eval_log_dir)
    utils.cleanup_log_dir(pre_eval_log_dir)

    # result_dir = expand_folder_dirs(args.result_dir, args)
    load_dir = expand_load_folder_dirs(args.load_dir, args)
    pretrain_dir = expand_save_folder_dirs(args.pretrain_dir, args)

    # Define environments variable
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.test_model == 'pretrained':
        log_dir = pre_log_dir
        eval_log_dir = pre_eval_log_dir

    # Define envs
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, args.gamma, log_dir, device, False)

    # Define obs_dim and act_dim
    obs_dim = int(envs.observation_space.shape[0])

    if len(envs.action_space.shape) > 0:
        act_dim = int(envs.action_space.shape[0])
    else:
        # act_dim = int(len([envs.action_space.sample()]))
        act_dim = envs.action_space.n

    print("************************************************************************************************")
    print("PRE-TRAINING ACTOR & POSTERIOR via Conditional VAE(CVAE)")
    print("0. task: ", args.env_name, " device: ", device)
    print("1. policy_lr & task_lr {}, b_lr: {}, discr_lr: {}, postr_lr: {}".format(args.lr, args.b_lr, args.d_lr, args.p_lr))

    # Load model(agent, posterior, discriminator)
    if args.load_model:
        if args.pretrain_algo == 'cvae':
            _agent, ob_rms, post, discr, trans = torch.load(os.path.join(load_dir, args.env_name + ".pt"))
            print("Load Trained Model from ", os.path.join(load_dir, args.env_name + ".pt"))
        elif args.pretrain_algo == 'bc':
            _agent, ob_rms = torch.load(os.path.join(load_dir, args.env_name + ".pt"))
            print("Load Trained Model from ", os.path.join(load_dir, args.env_name + ".pt"))

        actor_critic = _agent.actor_critic
        envs.ob_rms = ob_rms
        print("************************************************************************************************")
    else:
        # Define Policy and Task Transition
        if args.pretrain_algo == 'cvae':
            print("2. policy: hierarchical_policy")
            actor_critic = HierarchicalPolicy(
                envs.observation_space.shape,  # (#, )
                args.latent_dim,
                envs.action_space,  # Box(#, )
                base_kwargs={'recurrent': args.recurrent_policy, 'latent_size': args.latent_dim})
        elif args.pretrain_algo == 'bc':
            print("2. policy: standard_policy")
            actor_critic = Policy(
                envs.observation_space.shape,  # (#, )
                envs.action_space,  # Box(#, )
                base_kwargs={'recurrent': args.recurrent_policy})

        actor_critic.to(device)

        # Define Agent(A2C, PPO, ACKTR)
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

        # Define Posterior
        if args.posterior:
            print("4. posterior: true")
            post = posterior.Posterior(
                input_dim=obs_dim + act_dim,
                hidden_dim=100,
                latent_dim=args.latent_dim,
                p_lr=args.p_lr,
                action_space=envs.action_space,
                device=device)

        # Define Discriminator
        if args.gail:
            assert len(envs.observation_space.shape) == 1
            if args.gail_algo == 'variational':
                print("5. discriminator: variational_discriminator")
                discr = gail.VariationalDiscriminator(
                    input_dim=obs_dim + act_dim,
                    hidden_dim=100,
                    latent_dim=args.latent_dim,
                    init_beta=args.init_beta,
                    i_c=args.i_c,
                    d_lr=args.d_lr,
                    b_lr=args.b_lr,
                    device=device,
                    good_end=args.good_end)
            elif args.gail_algo == 'wasserstein':
                print("5. discriminator: wasserstein_discriminator")
                discr = gail.WassersteinDiscriminator(
                    input_dim=obs_dim + act_dim,
                    hidden_dim=100,
                    use_latent=args.use_latent,
                    d_lr=args.d_lr,
                    device=device,
                    good_end=args.good_end,
                    extract_obs=args.extract_obs)
            elif args.gail_algo == 'standard':
                print("5. discriminator: standard_discriminator")
                discr = gail.Discriminator(
                    input_dim=obs_dim + act_dim,
                    hidden_dim=100,
                    use_latent=args.use_latent,
                    d_lr=args.d_lr,
                    device=device,
                    good_end=args.good_end,
                    extract_obs=args.extract_obs,
                    action_space=envs.action_space)

        # Define Expert Demonstration File Path
        file_name = os.path.join(
            args.gail_experts_dir,
            "trajs_{}_{}.pt".format(args.env_name.split('-')[0].lower(), args.expert_algo))

        # Define DataLoader for Pre-Training
        pre_train_loader = torch.utils.data.DataLoader(
            expert_dataset.ExpertDataset(args.env_name, file_name),
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=True)

        # Define Discriminator
        if args.task_transition:
            print("6. task_transition_model: true")
            trans = task_trans_model.TaskTransitionModel(input_dim=obs_dim + args.latent_dim,
                                                         hidden_dim=100,
                                                         latent_dim=args.latent_dim,
                                                         init_beta=args.init_beta,
                                                         t_lr=args.t_lr,
                                                         b_lr=args.b_lr,
                                                         i_c=args.i_c,
                                                         device=device,
                                                         extract_obs=args.extract_obs,
                                                         action_space=envs.action_space,
                                                         is_recurrent=True)

            # Define DataLoader for Task Transition Model
            _all_trajectories = torch.load(file_name)
            task_batch_size = _all_trajectories['states'].size(1)  # (1500,v76, 2)

            task_train_loader = torch.utils.data.DataLoader(
                task_dataset.TaskDataset(args.env_name, file_name),
                batch_size=task_batch_size,
                shuffle=False,
                drop_last=True)

        # Define Conditional Variational Auto-Encoder(CVAE) and Behavior Cloning(BC)
        if args.pretrain_algo == 'cvae':
            c_vae = cvae.CVAE(agent, post, args.extract_obs, device, envs.action_space, lr=0.0001)
        elif args.pretrain_algo == 'bc':
            bcg = behavior_cloning.BehaviorCloning(agent, args.extract_obs, device, envs.action_space, lr=0.0001)
        print("************************************************************************************************")

    epochs = []
    max_rewards, min_rewards, median_rewards, mean_rewards = [], [], [], []

    # Training Loop
    for i in range(args.pretrain_epoch):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(agent.optimizer, i, args.pretrain_epoch,
                                         agent.optimizer.lr if args.algo == "acktr" else args.lr)

        if args.pretrain_algo == 'cvae':
            # Update CVAE
            for _ in range(args.cvae_epoch):
                reconstruct_loss, regularization_loss = c_vae.update(pre_train_loader,
                                                                     utils.get_vec_normalize(envs)._obfilt)
            # Update Task Transition Model
            for _ in range(args.task_epoch):
                task_loss, ib_loss, beta = trans.update(task_train_loader, post,
                                                        utils.get_vec_normalize(envs)._obfilt)
        elif args.pretrain_algo == 'bc':
            for _ in range(args.bc_epoch):
                bc_loss = bcg.update(pre_train_loader,
                                     utils.get_vec_normalize(envs)._obfilt)

        # print("mean: ", utils.get_vec_normalize(envs).ob_rms.mean, " var: ", utils.get_vec_normalize(envs).ob_rms.var)

        # evaluate trained model
        if args.eval_interval is not None and i % args.eval_interval == 0:
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            if args.pretrain_algo == 'cvae':
                episode_rewards = evaluate_with_latent(
                    actor_critic,
                    args.eval_episode,
                    args,
                    trans,
                    ob_rms, args.env_name, args.seed, args.num_processes, eval_log_dir, device)
            elif args.pretrain_algo == 'bc':
                episode_rewards = evaluate(
                    actor_critic,
                    args.eval_episode,
                    args,
                    ob_rms, args.env_name, args.seed, args.num_processes, eval_log_dir, device)

            epochs.append(i)
            mean_rewards.append(np.mean(episode_rewards))
            median_rewards.append(np.median(episode_rewards))
            min_rewards.append(np.min(episode_rewards))
            max_rewards.append(np.max(episode_rewards))

        # log for every interval-th episode
        if i % args.log_interval == 0:
            if args.pretrain_algo == 'cvae':
                bc_loss = 0.0
            elif args.pretrain_algo == 'bc':
                reconstruct_loss, regularization_loss, ib_loss, task_loss, beta = 0.0, 0.0, 0.0, 0.0, 0.0
            print(
                " Updates {} \n"
                " Last rollout:    reconstruct_loss {:>8.4f}\n"
                "               regularization_loss {:>8.4f}\n"
                "                           ib_loss {:>8.4f}\n"
                "                         task_loss {:>8.4f}\n"
                "                              beta {:>8.4f}\n"
                "                           bc_loss {:>8.4f}"
                    .format(i, reconstruct_loss, regularization_loss, ib_loss, task_loss, beta, bc_loss))
            print("===============================================================================================")

        # save for every interval-th episode or for the last epoch
        if (i % args.save_interval == 0 or i == args.pretrain_epoch - 1) and args.save_dir != "":
            if args.pretrain_algo == 'cvae':
                torch.save([
                    agent,
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None),
                    post, discr, trans],
                    os.path.join(pretrain_dir, args.env_name + "_" + str(i) + ".pt"))
            elif args.pretrain_algo == 'bc':
                torch.save([
                    agent,
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)],
                    os.path.join(pretrain_dir, args.env_name + "_" + str(i) + ".pt"))

            _epochs = np.reshape(np.array(epochs), (-1, 1))
            _mean_rewards = np.reshape(np.array(mean_rewards), (-1, 1))
            _median_rewards = np.reshape(np.array(median_rewards), (-1, 1))
            _min_rewards = np.reshape(np.array(min_rewards), (-1, 1))
            _max_rewards = np.reshape(np.array(max_rewards), (-1, 1))
            logs_data = np.hstack([_epochs, _mean_rewards, _median_rewards, _min_rewards, _max_rewards])

            np.savetxt(fname=os.path.join(log_dir, args.env_name.split('-')[0].lower() + "_" + str(i) + ".csv"),
                       X=logs_data, delimiter=',')

if __name__ == "__main__":
    pretrain()
