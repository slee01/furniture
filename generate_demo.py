import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import h5py
import torch
import gym

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

"""
python generate_demo.py --env-name MountainCar-v0 --algo ppo --episode 3000 --save-date 190906
python generate_demo.py --env-name Walker2d-v2 --algo acktr --episode 3000 --save-date 190909

"""

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--render',
    action='store_true',
    default=False,
    help='render simulator')
parser.add_argument(
    '--vis-interval',
    type=int,
    default=10,
    help='visualization interval, one save per n updates (default: 10)')
parser.add_argument(
    '--episode',
    type=int,
    default=100,
    help='the number of expert demonstrations  (default: 100')
parser.add_argument(
    '--env-name',
    default='PongNoFrameskip-v4',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--save-demo',
    action='store_true',
    default=False,
    help='save demonstration in h5py format')
parser.add_argument(
    '--save-date',
    default='190909',
    help='pt file name for saved agent model')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')

args = parser.parse_args()
args.det = not args.non_det
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_env = gym.make(args.env_name)
max_episode_steps = _env._max_episode_steps

env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1,
    None,
    None,
    device,
    allow_early_resets=False)

# Get a render function
render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
# actor_critic, ob_rms = torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))
if torch.cuda.is_available():
    agent, ob_rms = torch.load(os.path.join(args.load_dir, args.algo, args.save_date, args.env_name + ".pt"), map_location=device)
else:
    agent, ob_rms = torch.load(os.path.join(args.load_dir, args.algo, args.save_date, args.env_name + ".pt"), map_location=device)

# agent.actor_critic.to(device)
vec_norm = get_vec_normalize(env)

if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

# if args.render and render_func is not None:
#     render_func('human')

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

# Define states, actions, rewards, lens as list (convert that to a Numpy array later)
states, actions, rewards, lens = [], [], [], []

success = 0
if args.env_name == 'Hopper-v2':  # acktr-190907 -mac
    criteria = 3800
elif args.env_name == 'HalfCheetah-v2':  # ikostrikov-191002 -ubuntu
    criteria = 4500
elif args.env_name == 'Ant-v2':  # ikostrikov-191002 -ubuntu
    criteria = 4000
elif args.env_name == 'Walker2d-v2':  # acktr-190909(ikostrikov-191002: 6500) -mac & ubuntu
    criteria = 5800
elif args.env_name == 'Reacher-v2':  # acktr-190906 -mac
    criteria = -3.0
elif args.env_name == 'InvertedPendulum-v2':  # acktr-190906 -mac
    criteria = 1000.0
elif args.env_name == 'InvertedDoublePendulum-v2': # acktr-190906 -mac
    criteria = 9356.0
elif args.env_name == 'MountainCarContinuous-v0': # acktr-190906 -mac
    criteria = 93
elif args.env_name == 'MountainCar-v0': # ppo-191002(iterate #: 50) -mac
    criteria = -155.0
elif args.env_name == 'MountainGolfCar-v0': # ppo-190922 -mac
    criteria = -142.0
elif args.env_name == 'MountainToyCar-v0':  # ppo-190929 -mac
    criteria = -76.0
else:
    criteria = 0

for e in range(args.episode * 100):
    ess, eas, ers = [], [], []

    # obs = env.reset().to(device)
    # recurrent_hidden_states = torch.zeros(1, agent.actor_critic.recurrent_hidden_state_size).to(device)
    obs = env.reset()
    recurrent_hidden_states = torch.zeros(1, agent.actor_critic.recurrent_hidden_state_size)

    # masks = torch.zeros(1, 1).to(device)
    masks = torch.zeros(1, 1)
    episode_rewards = 0.0
    done = False

    while not done:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = agent.actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)

        # Obser reward and next obs
        # print("obs: ", obs, " act: ", action)

        next_obs, reward, done, _ = env.step(action)
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
                # render_func('human') # it is not applicable for custom_tasks
                env.render()

        ess.append(obs.cpu().numpy())
        eas.append(action.cpu().numpy())
        ers.append(reward.cpu().numpy())

        _obs = obs.cpu().numpy()
        obs = next_obs

    if np.sum(ers) >= criteria and len(ess) == max_episode_steps:
    # if np.sum(ers) >= criteria and len(ess) == 140: # for MountainGolfCar
    # if np.sum(ers) >= criteria and len(ess) == 67: # for MountainCarContinuous
        # h5py format file should take input as same length array
        # Unless you get the error like below
        # TypeError: Object dtype dtype('O') has no native HDF5 equivalent

    # if (np.sum(ers) > criteria) and (np.sum(ers) < criteria+10):
    #     while(1):
    #         ess.append(_obs)
    #         eas.append(action.cpu().numpy())
    #         ers.append(reward.cpu().numpy())
    #         print("obs: ", ess[-1], " act: ", eas[-1])
    #
    #         if len(ess) == abs(criteria):
    #             break

        success += 1

        states.append(ess)
        actions.append(eas)
        rewards.append(ers)
        lens.append(len(ess))

    print("episode: ", e, " length: ", len(ess), " returns: ", episode_rewards, " success: ", success)

    if success >= args.episode:
        break

########################################################################################################################
########################################### CONVERT TRAJS TO H5PY FILE #################################################
########################################################################################################################
states, actions, rewards, lens = np.array(states), np.array(actions), np.array(rewards), np.array(lens)

assert len(states) == len(actions), 'len(states) != len(actions)'
assert len(states) == len(rewards), 'len(states) != len(rewards)'
assert len(states) == len(lens), 'len(states) != len(lens)'

if torch.cuda.is_available():
    save_dir = '/home/slee01/PycharmProjects/pytorch-a2c-ppo-acktr-gail/gail_experts/'
else:
    save_dir = '/Users/slee01/PycharmProjects/pytorch-a2c-ppo-acktr-gail/gail_experts/'

# save_path = os.path.join(save_dir, args.env_name + "_" + args.seed + ".h5")
save_path = os.path.join(
                save_dir,
                "trajs_{}_{}.h5".format(args.env_name.split('-')[0].lower(), args.algo))

print("=================================================================")
print("env_name: ", args.env_name)
print("seed for test: ", args.seed)
print("saved file name: ", save_path)

if args.save_demo:
    h5f = h5py.File(save_path, 'w')

    h5f.create_dataset('obs_B_T_Do', data=states)
    h5f.create_dataset('a_B_T_Da', data=actions)
    h5f.create_dataset('r_B_T', data=rewards)
    h5f.create_dataset('len_B', data=lens)

    key_list = list(h5f.keys())

    h5f.close()

    print("saved file keys: ", key_list)

print("expert_states: ", states.shape)
print("expert_actions: ", actions.shape)
print("expert_rewards: ", rewards.shape)
print("expert_lens: ", lens.shape)
# expert_states:  (10, 1000, 1, 17)
# expert_actions:  (10, 1000, 1, 6)
# expert_rewards:  (10, 1000, 1, 1)
# expert_lens:  (10,)
print("=================================================================")

