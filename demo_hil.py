"""
How to use the IKEA furniture assembly environment with gym interface.
For parallel execution, set `--num_envs` flag to larger than 1.
"""

# site-packages
import time
import gym
import imageio
import os
import time
import random
import copy
import torch
import numpy as np

from collections import deque

# a2c_ppo_acktr
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
# from evaluation import evaluate, evaluate_with_latent

# furniture
#from env import make_vec_env

np.set_printoptions(precision=4)
np.set_printoptions(threshold=10)

def main(config):
    """
    Makes a gym environment with name @env_id and runs one episode.
    If @config.num_env is larger than 1, it will use SubprocVecEnv to run
    multiple environments in parallel.
    """
    num_env = config.num_env
    env_id = config.env_id

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cuda:0":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # create a gym environment instance
    assert num_env == 1, "TODO: multiple envs do NOT work yet"
    env = gym.make(env_id, **config.__dict__)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    args = get_args()
    agent, actor_critic, post, discr, trans = get_model(args, env, obs_dim, act_dim, device)

    print("************************************************************************************************")
    print("0. task: ", args.env_name, " device: ", device)
    print("   observation shape: ", env.observation_space.shape)
    print("1. policy_lr & task_lr {}, b_lr: {}, discr_lr: {}, postr_lr: {}".format(args.lr, args.b_lr, args.d_lr, args.p_lr))

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
                                        env.observation_space.shape, env.action_space, args.latent_dim,
                                        # actor_critic.recurrent_hidden_state_size, args.latent_dim * 4)
                                        args.latent_space, actor_critic.recurrent_hidden_state_size, trans.hidden_dim)
    else:
        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                  env.observation_space.shape, env.action_space,
                                  actor_critic.recurrent_hidden_state_size)
    obs = env.reset().to(device)
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

        for step in range(args.num_steps):
            with torch.no_grad():
                if args.hierarchical_policy and args.task_transition:
                    # task, task_features, recurrent_hidden_task_states = trans.act(
                    task, task_features = trans.act(
                        rollouts.obs[step],
                        prev_task,
                        # rollouts.recurrent_hidden_task_states[step],
                        # rollouts.masks[step],
                        mean_mode=False,
                        use_random_latent=args.use_random_latent,
                        use_constant_latent=args.use_constant_latent,
                        constant_latent=args.constant_latent)

                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        task,
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step],
                        deterministic=False)
                else:
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step], rollouts.masks[step],
                        deterministic=False)

            obs, reward, done, infos = env.step(action.cpu())

            if args.render and j % args.vis_interval == 0:
                env.render()

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

            if args.hierarchical_policy:
                # rollouts.insert(obs, task, prev_task, recurrent_hidden_states, recurrent_hidden_task_states, action,
                rollouts.insert(obs, task, prev_task, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)
            else:
                rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)

            if args.hierarchical_policy:
                for i in range(len(done)):
                    if done[i] and args.latent_space == 'continuous':
                        task[i] = torch.normal(mean=torch.zeros(1, args.latent_dim), std=torch.ones(1, args.latent_dim))
                    elif done[i] and args.latent_space == 'discrete':
                        task[i] = torch.Tensor(random.sample(range(args.latent_dim), 1)).to(device)
                prev_task = task

        mean_rewards.append(np.mean(episode_rewards))
        median_rewards.append(np.median(episode_rewards))
        min_rewards.append(np.min(episode_rewards))
        max_rewards.append(np.max(episode_rewards))

        raw_rewards = copy.deepcopy(rollouts.rewards.view(1,-1))

        if args.posterior:
            post_epoch = args.posterior_epoch

            for _ in range(post_epoch):
                pos_loss = post.update(rollouts, args.posterior_batch_size)

            for step in range(args.num_steps):
                rollouts.rewards[step] = args.posterior_reward_coef * post.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], rollouts.tasks[step],
                    args.gamma, rollouts.masks[step])

            for step in range(args.num_steps):
                rollouts.tasks[step] = post.relabel_task(
                    rollouts.obs[step], rollouts.actions[step])
                if step is not 0:
                    rollouts.prev_tasks[step] = rollouts.tasks[step-1]

        post_rewards = copy.deepcopy(rollouts.rewards.view(1, -1) - raw_rewards)

        if args.gail:
            gail_epoch = args.gail_epoch

            if j >= 10:
                env.venv.eval()
            elif env.action_space.__class__.__name__ is not "Discrete":
                gail_epoch = 100

            for _ in range(gail_epoch):
                dis_loss, gail_loss, grad_loss = discr.update(post, gail_train_loader, rollouts,
                                                              utils.get_vec_normalize(env)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] += args.discr_reward_coef * discr.predict_reward(
                    rollouts.obs[step], rollouts.tasks[step], rollouts.actions[step],
                    args.gamma, rollouts.masks[step])

        discr_rewards = copy.deepcopy(rollouts.rewards.view(1, -1) - post_rewards - raw_rewards)

        if args.task_transition:
            if args.task_curiosity_reward:
                for step in range(args.num_steps):
                    rollouts.rewards[step] += args.task_reward_coef * trans.predict_reward(
                        rollouts.obs[step], rollouts.prev_tasks[step], rollouts.tasks[step],
                        args.gamma, rollouts.masks[step])

            task_epoch = args.task_epoch

            for _ in range(task_epoch):
                if args.fix_beta:
                    trans.beta = 0.0
                task_loss, ib_loss, beta = trans.update(rollouts, agent, utils.get_vec_normalize(env)._obfilt)

        task_rewards = copy.deepcopy(rollouts.rewards.view(1, -1) - post_rewards - discr_rewards - raw_rewards)

        print("-----------------------------------------------------------------------------------------------")
        print("   raw_rewards: ", raw_rewards.cpu().numpy())
        print(" postr_rewards: ", post_rewards.cpu().numpy())
        print(" discr_rewards: ", discr_rewards.cpu().numpy())
        print(" task_rewards: ", task_rewards.cpu().numpy())
        print(" final_rewards: ", rollouts.rewards.view(1, -1).cpu().numpy())
        print("-----------------------------------------------------------------------------------------------")

        with torch.no_grad():
            if args.hierarchical_policy:
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.tasks[-1],
                    rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]).detach()
            else:
                next_value = actor_critic.get_value(
                    rollouts.obs[-1],
                    rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        if args.hierarchical_policy:
            value_loss, action_loss, dist_entropy, bc_loss = agent.update(policy_train_loader, rollouts, post,
                                                                          utils.get_vec_normalize(env)._obfilt)
        else:
            value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # if args.eval_interval is not None and j % args.eval_interval == 0:
        #     eval_epochs.append(j)
        #     ob_rms = utils.get_vec_normalize(env).ob_rms
        #
        #     if args.hierarchical_policy:
        #         _episode_rewards = evaluate_with_latent(
        #             actor_critic, args.eval_episode, args, trans,
        #             ob_rms, args.env_name, args.seed, args.num_processes, eval_log_dir, device)
        #     else:
        #         _episode_rewards = evaluate(
        #             actor_critic, args.eval_episode, args,
        #             ob_rms, args.env_name, args.seed, args.num_processes, eval_log_dir, device)
        #
        #     eval_mean_rewards.append(np.mean(_episode_rewards))
        #     eval_median_rewards.append(np.median(_episode_rewards))
        #     eval_min_rewards.append(np.min(_episode_rewards))
        #     eval_max_rewards.append(np.max(_episode_rewards))
        #
        # if (j % args.save_interval == 0 or j == num_updates - 1) and args.save_dir != "":
        #     _epochs = np.reshape(np.array(epochs), (-1, 1))
        #     _mean_rewards = np.reshape(np.array(mean_rewards), (-1, 1))
        #     _median_rewards = np.reshape(np.array(median_rewards), (-1, 1))
        #     _min_rewards = np.reshape(np.array(min_rewards), (-1, 1))
        #     _max_rewards = np.reshape(np.array(max_rewards), (-1, 1))
        #
        #     _eval_epochs = np.reshape(np.array(eval_epochs), (-1, 1))
        #     _eval_mean_rewards = np.reshape(np.array(eval_mean_rewards), (-1, 1))
        #     _eval_median_rewards = np.reshape(np.array(eval_median_rewards), (-1, 1))
        #     _eval_min_rewards = np.reshape(np.array(eval_min_rewards), (-1, 1))
        #     _eval_max_rewards = np.reshape(np.array(eval_max_rewards), (-1, 1))
        #
        #     logs_data = np.hstack([_epochs,
        #                            _mean_rewards,
        #                            _median_rewards,
        #                            _min_rewards,
        #                            _max_rewards])
        #
        #     eval_logs_data = np.hstack([_eval_epochs,
        #                                 _eval_mean_rewards,
        #                                 _eval_median_rewards,
        #                                 _eval_min_rewards,
        #                                 _eval_max_rewards])
        #
        #     np.savetxt(fname=os.path.join(log_dir, args.env_name.split('-')[0].lower() + "_" + str(j) + ".csv"),
        #                X=logs_data, delimiter=',')
        #     np.savetxt(fname=os.path.join(eval_log_dir, args.env_name.split('-')[0].lower() + "_" + str(j) + ".csv"),
        #                X=eval_logs_data, delimiter=',')
        #
        #     if args.hierarchical_policy:
        #         torch.save([
        #             agent,
        #             getattr(utils.get_vec_normalize(env), 'ob_rms', None),
        #             post, discr, trans
        #         ], os.path.join(save_dir, args.env_name + "_" + str(j) + ".pt"))
        #
        #         _best_mean_rewards = evaluate_with_latent(
        #             actor_critic, args.save_episode, args, trans,
        #             ob_rms, args.env_name, args.seed, args.num_processes, eval_log_dir, device)
        #
        #         if np.mean(_best_mean_rewards) > best_mean_rewards:
        #             best_mean_rewards = np.mean(_best_mean_rewards)
        #             torch.save([
        #                 agent,
        #                 getattr(utils.get_vec_normalize(env), 'ob_rms', None),
        #                 post, discr, trans
        #             ], os.path.join(save_dir, args.env_name + ".pt"))
        #     else:
        #         torch.save([
        #             agent,
        #             getattr(utils.get_vec_normalize(env), 'ob_rms', None)
        #         ], os.path.join(save_dir, args.env_name + "_" + str(j) + ".pt"))
        #
        #         _best_mean_rewards = evaluate(
        #             actor_critic, args.save_episode, args,
        #             ob_rms, args.env_name, args.seed, args.num_processes, eval_log_dir, device)
        #
        #         if np.mean(_best_mean_rewards) > best_mean_rewards:
        #             best_mean_rewards = np.mean(_best_mean_rewards)
        #             torch.save([
        #                 agent,
        #                 getattr(utils.get_vec_normalize(env), 'ob_rms', None),
        #                 post, discr, trans
        #             ], os.path.join(save_dir, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()

            if args.hierarchical_policy is False:
                ib_loss, task_loss, beta, pos_loss, bc_loss = 0.0, 0.0, 0.0, 0.0, 0.0
            if args.gail is False:
                dis_loss = 0.0

            print("===============================================================================================")
            print(
                " Updates {}, num timesteps {}, FPS {} \n"
                " Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                " Last rollout:   value_loss {:>8.4f}\n"
                "                action_loss {:>8.4f}\n"
                "                    bc_loss {:>8.4f}\n"
                "               dist_entropy {:>8.4f}\n"
                "         discriminator_loss {:>8.4f}\n"
                "                  gail_loss {:>8.4f}\n"
                "                  grad_loss {:>8.4f}\n"
                "                    ib_loss {:>8.4f}\n"
                "                  task_loss {:>8.4f}\n"
                "                       beta {:>8.4f}\n"

                "             posterior_loss {:>8.4f}"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), value_loss, action_loss, bc_loss,
                        dist_entropy, dis_loss, gail_loss, grad_loss, ib_loss, task_loss, beta, pos_loss))
            print("===============================================================================================")

    # # reset the environment
    # observation = env.reset()
    # done = False
    #
    # st = time.time()
    # for i in range(500):
    #     # take a step with a randomly sampled action
    #     observation, reward, done, info = env.step(env.action_space.sample())
    # print('FPS = {}'.format(500 / (time.time() - st)))
    #
    # # close the environment instance
    # env.close()


def get_model(args, env, obs_dim, act_dim, device):
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

    log_dir = expand_save_folder_dirs(args.log_dir, args)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(eval_log_dir)

    save_dir = expand_save_folder_dirs(args.save_dir, args)
    pretrain_dir = expand_load_folder_dirs(args.pretrain_dir, args)

    if args.load_model:
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
                max_grad_norm=args.max_grad_norm)

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
            assert len(env.observation_space.shape) == 1
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
                    action_space=env.action_space)
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
                    action_space=env.action_space)

        if args.posterior and args.reset_posterior:
            print("7. reset the posterior network: true")
            post = posterior.Posterior(
                input_dim=obs_dim + act_dim,
                hidden_dim=100,
                feature_dim=32,
                latent_dim=args.latent_dim,
                latent_space=args.latent_space,
                p_lr=args.p_lr,
                action_space=env.action_space,
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
                                                         action_space=env.action_space,
                                                         is_recurrent=True)

        reset_model_machines(agent.actor_critic, device)
        reset_model_machines(post, device)
        reset_model_machines(discr, device)
        reset_model_machines(trans, device)

        trans.i_c = args.i_c
        actor_critic = agent.actor_critic
        env.ob_rms = ob_rms
        print("   information_botlleneck constraint: ", trans.i_c)
        print("************************************************************************************************")
    else:
        if args.hierarchical_policy:
            print("2. policy: hierarchical_policy")
            actor_critic = HierarchicalPolicy(
                env.observation_space.shape,
                args.latent_dim,
                args.latent_space,
                env.action_space,
                base_kwargs={'recurrent': args.recurrent_policy, 'latent_size': args.latent_dim})
        else:
            print("2. policy: standard_policy")
            actor_critic = Policy(
                env.observation_space.shape,
                env.action_space,
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
                action_space=env.action_space,
                device=device)

        if args.gail:
            assert len(env.observation_space.shape) == 1
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
                    action_space=env.action_space)
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
                    action_space=env.action_space)

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
                                                         action_space=env.action_space,
                                                         is_recurrent=True)

    return agent, actor_critic, post, discr, trans

def argsparser():
    """
    Returns argument parser for furniture assembly environment.
    """
    import argparse
    from util import str2bool

    parser = argparse.ArgumentParser("Demo for IKEA Furniture Assembly Environment")
    parser.add_argument('--env_id', type=str, default='furniture-baxter-v0')
    parser.add_argument('--num_env', type=int, default=1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--debug', type=str2bool, default=False)

    import config.furniture as furniture_config
    furniture_config.add_argument(parser)

    parser.set_defaults(visual_ob=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argsparser()
    main(args)
