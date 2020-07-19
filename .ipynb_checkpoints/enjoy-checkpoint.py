import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import h5py
import torch
import gym

from scipy import sparse

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

"""
<TEST CVAE pre-training>
python enjoy.py --env-name FetchPickAndPlace-v1 --algo ppo --episode 10 --save-date 191023_CVAE \
--use-latent --latent-dim 1 --task-transition --render --vis-interval 1


python enjoy.py --env-name InvertedPendulum-v2 --algo acktr --episode 10 --save-date 190909_hGAIL

python enjoy.py --env-name MountainGolfCar-v1 --algo ppo --episode 10 --save-date 190922_hGAIL \
--use-latent --latent-dim 1 --render --vis-interval 1

python enjoy.py --env-name MountainToyCar-v1 --algo ppo --episode 10 --save-date 190928_wasserstein_GAIL_with_latent_p15 \
--use-latent --latent-dim 1 --render --vis-interval 1

python enjoy.py --env-name FetchPickAndPlace-v1 --algo ppo --episode 10 \
--save-date 191023_wasserstein_GAIL_with_latent_1_p13 --use-latent --latent-dim 1 --task-transition --render --vis-interval 1

python enjoy.py --env-name FetchPickAndPlace-v1 --algo a2c --episode 10 \
--save-date 191023_CAVE --use-latent --latent-dim 1 --task-transition --render --vis-interval 1

# MountainToyCar Success Command
python generate_latent_graph.py --env-name MountainOldCar-v1 --algo ppo --episode 10 \
--save-date 191006_standard_GAIL_with_latent_1_p10_v2 --use-latent --latent-dim 1 --task-transition

"""

sys.path.append('a2c_ppo_acktr')

def reset_model_machines(model, device):
    model.to(device)
    model.device = device

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

args = get_args()
# args.det = not args.non_det
# print("deterministic: ", args.det)

# Define directories
log_dir = expand_save_folder_dirs(args.log_dir, args)
pre_log_dir = expand_save_folder_dirs(args.pre_log_dir, args)
eval_log_dir = log_dir + "_eval"
pre_eval_log_dir = eval_log_dir + "_eval"
utils.cleanup_log_dir(eval_log_dir)

result_dir = expand_save_folder_dirs(args.result_dir, args)
pre_result_dir = expand_save_folder_dirs(args.pre_result_dir, args)
load_dir = expand_load_folder_dirs(args.load_dir, args)
pre_load_dir = expand_load_folder_dirs(args.pre_load_dir, args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if args.test_model == 'pretrained':
    log_dir = pre_log_dir
    eval_log_dir = pre_eval_log_dir
    result_dir = pre_result_dir
    load_dir = pre_load_dir

print("=================================================================")
print("0. env_name: ", args.env_name)
print("1. seed for test: ", args.seed)
print("2. log directory: ", log_dir)

# We need to use the same statistics for normalization as used in training
if torch.cuda.is_available():
    if args.use_latent:
        agent, ob_rms, post, discr, trans = torch.load(os.path.join(load_dir, args.env_name + ".pt"))
        print("4. load directory: ", os.path.join(load_dir, args.env_name + ".pt"))

        reset_model_machines(post, device)
        reset_model_machines(discr, device)
        reset_model_machines(trans, device)

    else:
        agent, ob_rms = torch.load(os.path.join(load_dir, args.env_name + ".pt"))
        print("4. load directory: ", os.path.join(load_dir, args.env_name + ".pt"))

    reset_model_machines(agent.actor_critic, device)

else:
    if args.use_latent:
        agent, ob_rms, post, discr, trans = torch.load(os.path.join(load_dir, args.env_name + ".pt"), map_location='cpu')
        print("4. load directory: ", os.path.join(load_dir, args.env_name + ".pt"))

        reset_model_machines(post, device)
        reset_model_machines(discr, device)
        reset_model_machines(trans, device)

    else:
        agent, ob_rms = torch.load(os.path.join(load_dir, args.env_name + ".pt"), map_location='cpu')
        print("4. load directory: ", os.path.join(load_dir, args.env_name + ".pt"))

    reset_model_machines(agent.actor_critic, device)

env = make_vec_envs(args, args.env_name, args.seed + 1000, 1, None, None,
                    device='cpu', allow_early_resets=False)

# Get a render function and vectorize environments
render_func = get_render_func(env)
vec_norm = get_vec_normalize(env)

# if args.render and render_func is not None:
    # render_func('human')  # it is not applicable for custom_tasks

if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i


print("=================================================================")
if args.save_result is False:
    print("DO NOT SAVE RESULTS")

dones = []
episode_returns = []
# latent variable evaluation for each episode
for e in range(args.episode):
    states, actions, rewards, task_mus, task_sigmas, task_logits, tasks = [], [], [], [], [], [], []

    obs = env.reset().to(device)
    recurrent_hidden_states = torch.zeros(1, agent.actor_critic.recurrent_hidden_state_size).to(device)

    if args.use_latent and args.task_transition:
        # prev_task = torch.normal(
        #     mean=torch.zeros(args.num_processes, args.latent_dim),
        #     std=torch.ones(args.num_processes, args.latent_dim)).to(device)
        prev_task = torch.randn((args.num_processes, args.latent_dim)).to(device)
        prev_task = torch.zeros_like(prev_task).to(device)
        recurrent_hidden_task_states = torch.zeros(1, args.latent_dim * 4).to(device)

    episode_rewards = 0.0
    masks = torch.zeros(1, 1).to(device)
    done = False

    while not done:
        with torch.no_grad():
            if args.use_latent and args.task_transition:  # recurrent_hidden_task_states, masks,
#                 task, task_feature, recurrent_hidden_task_states = trans.act(obs, prev_task,
#                                                                              mean_mode=True,
#                                                                              use_random_latent=args.use_random_latent,
#                                                                              use_constant_latent=args.use_constant_latent,
#                                                                              constant_latent=args.constant_latent)
                task, task_feature = trans.act(obs, prev_task,
                                               mean_mode=True,
                                               use_random_latent=args.use_random_latent,
                                               use_constant_latent=args.use_constant_latent,
                                               constant_latent=args.constant_latent)
    
                if args.latent_space == 'continuous':
                    task_mu, task_sigma = trans.get_dist_params(task_feature)
                elif args.latent_space == 'discrete':
                    task_logit = trans.get_dist_params(task_feature)

                value, action, action_log_prob, recurrent_hidden_states = agent.actor_critic.act(
                    # obs, task, recurrent_hidden_states, masks, deterministic=args.det)
                    obs, task, recurrent_hidden_states, masks, deterministic=True)
            else:
                value, action, action_log_prob, recurrent_hidden_states = agent.actor_critic.act(
                    # obs, recurrent_hidden_states, masks, deterministic=args.det)
                    obs, recurrent_hidden_states, masks, deterministic=True)

        next_obs, reward, done, infos = env.step(action.cpu())

        episode_rewards += reward
        masks.fill_(0.0 if done else 1.0)

        if args.env_name.find('Bullet') > -1:
            if torsoId > -1:
                distance = 5
                yaw = 0
                humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
                p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

        if render_func is not None:
            if args.render and e % args.vis_interval == 0:
                env.render()

        states.append(obs.cpu().numpy())
        actions.append(action.cpu().numpy())
        rewards.append(reward.cpu().numpy())

        if args.use_latent and args.task_transition:
            tasks.append(task.cpu().numpy())
            if args.latent_space == 'continuous':
                task_mus.append(task_mu.cpu().numpy())
                task_sigmas.append(task_sigma.cpu().numpy())
            elif args.latent_space == 'discrete':
                task_logits.append(task_logit.cpu().numpy())

            prev_task = task

        obs = next_obs.to(device)

    if args.env_name == "FetchPickAndPlace-v1":
        print("env_name: FetchPickAndPlace-v1")
        if abs(reward) < 0.1:
            done = True
        else:
            done = False
    elif args.env_name == "MountainToyCar-v1":
        print("env_name: MountainToyCar-v1")
        if len(states) < 200:
            done = True
        else:
            done = False
    elif args.env_name == "MountainToyCarContinuous-v1":
        print("env_name: MountainToyCarContinuous-v1")
        if abs(reward) > 0.0:
            done = True
        else:
            done = False

    dones.append(done)
    episode_returns.append(episode_rewards)
    print("episode: ", e, " length: ", len(states), " returns: ", episode_rewards, " result: ", abs(reward), " done: ", done)

    if args.save_result:
        if args.use_latent and args.task_transition:
            task_mus, task_sigmas, task_logits, tasks = np.array(task_mus), np.array(task_sigmas), np.array(task_logits), np.array(tasks)
            if args.latent_space == 'continuous':
                task_mus, task_sigmas, tasks = np.squeeze(task_mus, axis=1), np.squeeze(task_sigmas, axis=1), np.squeeze(tasks, axis=1)
            elif args.latent_space == 'discrete':
                task_logits, tasks = np.squeeze(task_logits, axis=1), np.squeeze(tasks, axis=1)

            if args.latent_space == 'continuous':
                assert len(tasks) == len(task_mus), 'len(states) != len(actions)'
                assert len(tasks) == len(task_sigmas), 'len(states) != len(rewards)'
            elif args.latent_space == 'discrete':
                assert len(tasks) == len(task_logits), 'len(tasks) != len(rewards)'
            assert len(tasks) == len(rewards), 'len(tasks) != len(rewards)'

        states = np.squeeze(np.array(states), axis=1)
        actions = np.squeeze(np.array(actions), axis=1)
        rewards = np.squeeze(np.array(rewards), axis=1)
        print(tasks.shape, states.shape, actions.shape, rewards.shape)


        if args.use_latent and args.task_transition:
            if args.latent_space == 'continuous':
#                 task_data = np.hstack([states, actions, rewards, task_mus, task_sigmas, tasks])
                task_data = np.hstack([states, actions, rewards, task_mus, task_sigmas, tasks])
            elif args.latent_space == 'discrete':
                task_data = np.hstack([states, actions, rewards, task_logits, tasks])
        else:
            task_data = np.hstack([states, actions, rewards])

        np.savetxt(fname=os.path.join(result_dir, args.env_name.split('-')[0].lower() + "_" + str(e) + ".csv"),
                   X=task_data, delimiter=',')
        print("save file: ", os.path.join(result_dir, args.env_name.split('-')[0].lower() + "_" + str(e) + ".csv"))


dones = np.array(dones)
success_rate = np.sum(dones) / dones.shape[0]
print("=========================================================")
print("env_name:                   ", args.env_name)
print("load_model:                       ", load_dir)
print("mean and variance of episode_returns: ", np.mean(episode_returns), np.std(episode_returns))
print("success_rate:               ", success_rate)
print("=========================================================")
