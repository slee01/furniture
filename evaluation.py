import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs


def evaluate(actor_critic, num_episodes, args, ob_rms, env_name, seed, num_processes, eval_log_dir, device):

    if num_episodes == args.save_interval:
        eval_envs = make_vec_envs(env_name, seed + 1000, num_processes, None, eval_log_dir, device, True)
    else:
        eval_envs = make_vec_envs(env_name, seed+num_processes, num_processes, None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < num_episodes:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # obs, _, done, infos = eval_envs.step(torch.squeeze(action))
        obs, _, done, infos = eval_envs.step(action.cpu())

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    return eval_episode_rewards


def evaluate_with_latent(actor_critic, num_episodes, args, trans, ob_rms, env_name, seed, num_processes, eval_log_dir, device):

    if num_episodes == args.save_interval:
        eval_envs = make_vec_envs(args, env_name, seed + 1000, num_processes, None, eval_log_dir, device, True)
    else:
        eval_envs = make_vec_envs(args, env_name, seed+num_processes, num_processes, None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)

    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []
    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    # eval_task = torch.randn((num_processes, args.latent_dim)).to(device)
    if args.latent_space == 'continuous':
        eval_task = torch.randn(size=(num_processes, args.latent_dim)).to(device)
    elif args.latent_space == 'discrete':
        eval_task = torch.randint(high=args.latent_dim, size=(num_processes, args.latent_dim)).to(device)

    # eval_recurrent_hidden_task_states = torch.zeros(1, trans.hidden_dim).to(device)

    while len(eval_episode_rewards) < num_episodes:
        with torch.no_grad():
            # eval_task, eval_task_features, eval_recurrent_hidden_task_states = \
            eval_task, eval_task_features = trans.act(obs, eval_task,
                                                      # eval_recurrent_hidden_task_states,
                                                      #  eval_masks,
                                                      mean_mode=True,
                                                      use_random_latent=args.use_random_latent,
                                                      use_constant_latent=args.use_constant_latent,
                                                      constant_latent=args.constant_latent)

            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_task,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # obs, _, done, infos = eval_envs.step(torch.squeeze(action))
        obs, _, done, infos = eval_envs.step(action.cpu())

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    return eval_episode_rewards
