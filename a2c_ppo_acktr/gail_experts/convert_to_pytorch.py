import argparse
import os
import sys
import pickle
import glob

import numpy as np
import torch

"""
python convert_to_pytorch.py --h5-file trajs_halfcheetah.h5
python convert_to_pytorch.py --pkl-file Baxter_0.pkl | Cursor_7.pkl | Saywer_7.pkl
"""


def get_demo_files(demo_file_path):
    demos = []
    for f in glob.glob(demo_file_path + "_*"):
        if os.path.isfile(f):
            demos.append(f)
    return demos

def main():
    parser = argparse.ArgumentParser(
        'Converts expert trajectories from h5 to pt format.')
    parser.add_argument(
        '--pkl-file',
        default='demos/Baxter_toy_table',
        help='input pkl file',
        type=str)
    parser.add_argument(
        '--pt-file',
        default=None,
        help='output pt file, by default replaces file extension with pt',
        type=str)
    args = parser.parse_args()

    # if args.pt_file is None:
    #     args.pt_file = os.path.splitext(args.pkl_file)[0] + '.pt'

    assert (
            args.pkl_file is not None
    ), "--demo_path should be set (e.g. demos/Sawyer_toy_table)"
    demo_files = get_demo_files(args.pkl_file)

    print("file_path: ", args.pkl_file)
    print("demo_files: ", demo_files)

    statesArray, actionsArray, rewardsArray, lenArray = [], [], [], []

    # now load the picked numpy arrays
    for file_path in demo_files:
        states, actions, rewards = [], [], []

        with open(file_path, "rb") as f:
            demo = pickle.load(f)

            print("demo: ", type(demo), demo.keys())
            print("demo.qpos: ", type(demo['qpos']))
            for i in range(len(demo['qpos'])):
                print("i: ", i, demo['qpos'][i])
            print("demo.actions: ", type(demo['actions']))
            # add observations
            for state in demo["obs"]:
                states.append(state)
            states.pop()

            # add actions
            for action in demo["actions"]:
                actions.append(action)

            # add rewards
            if "rewards" in demo:
                for reward in demo["rewards"]:
                        rewards.append(reward)

            dataset_size = len(states)
            print("data: ", args.pkl_file)
            print("dataset_size: ",  dataset_size)
            print("states: ", np.array(states).shape)
            print("actions: ", np.array(actions).shape)
            print("rewards: ", np.array(rewards).shape)

        statesArray.append(states)
        actionsArray.append(actions)
        rewardsArray.append(rewards)
        lenArray.append(len(states))

    # with h5py.File(args.pkl_file, 'r') as f:
    #     dataset_size = f['obs_B_T_Do'].shape[0]  # full dataset size
    #
    #     states = f['obs_B_T_Do'][:dataset_size, ...][...]
    #     actions = f['a_B_T_Da'][:dataset_size, ...][...]
    #     rewards = f['r_B_T'][:dataset_size, ...][...]
    #     lens = f['len_B'][:dataset_size, ...][...]
    #
    #     print("rewards: ", rewards.shape)
    #     for i in range(dataset_size):
    #         print(i, " rewards: ", np.sum(rewards[i]))
    #
    #     # (200, 1000, 1, 1) vs. (200, 1000,
    #     if len(states.shape) >= 4:
    #         print("states: ", states.shape)
    #         states = np.squeeze(states, axis=2)
    #         # states = np.squeeze(states)
    #         print("states: ", states.shape)
    #     if len(actions.shape) >= 4:
    #         print("actions: ", actions.shape)
    #         actions = np.squeeze(actions, axis=2)
    #         # actions = np.squeeze(actions)
    #         print("actions: ", actions.shape)
    #     if len(rewards.shape) >= 4:
    #         print("rewards: ", rewards.shape)
    #         rewards = np.squeeze(rewards, axis=2)
    #         # rewards = np.squeeze(rewards)
    #         print("rewards: ", rewards.shape)
    #     if len(lens.shape) >= 2:
    #         print("lens: ", lens.shape)
    #         lens = np.squeeze(lens, axis=1)
    #         # lens = np.squeeze(lens)
    #         print("lens: ", lens.shape)
    #
    #     states = torch.from_numpy(states).float()
    #     actions = torch.from_numpy(actions).float()
    #     rewards = torch.from_numpy(rewards).float()
    #     lens = torch.from_numpy(lens).long()

    # (trajs, episodes, features)
    # states: torch.Size([3000, 50, 28])
    # actions: torch.Size([3000, 50, 4])
    # rewards: torch.Size([3000, 50, 1])
    # lens: torch.Size([3000])

    data = {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'lengths': lens
    }

    torch.save(data, args.pt_file)


if __name__ == '__main__':
    main()
