import argparse
import os
import sys

import h5py
import numpy as np
import torch

"""
python convert_to_pytorch.py --h5-file trajs_halfcheetah.h5
python convert_to_pytorch.py --pkl-file Baxter_0.pkl | Cursor_7.pkl | Saywer_7.pkl
"""

def main():
    parser = argparse.ArgumentParser(
        'Converts expert trajectories from h5 to pt format.')
    parser.add_argument(
        '--pkl-file',
        default='Baxter_0.pkl',
        help='input pkl file',
        type=str)
    parser.add_argument(
        '--pt-file',
        default=None,
        help='output pt file, by default replaces file extension with pt',
        type=str)
    args = parser.parse_args()

    if args.pt_file is None:
        args.pt_file = os.path.splitext(args.h5_file)[0] + '.pt'

    with h5py.File(args.h5_file, 'r') as f:
        dataset_size = f['obs_B_T_Do'].shape[0]  # full dataset size

        states = f['obs_B_T_Do'][:dataset_size, ...][...]
        actions = f['a_B_T_Da'][:dataset_size, ...][...]
        rewards = f['r_B_T'][:dataset_size, ...][...]
        lens = f['len_B'][:dataset_size, ...][...]

        print("rewards: ", rewards.shape)
        for i in range(dataset_size):
            print(i, " rewards: ", np.sum(rewards[i]))

        # (200, 1000, 1, 1) vs. (200, 1000,
        if len(states.shape) >= 4:
            print("states: ", states.shape)
            states = np.squeeze(states, axis=2)
            # states = np.squeeze(states)
            print("states: ", states.shape)
        if len(actions.shape) >= 4:
            print("actions: ", actions.shape)
            actions = np.squeeze(actions, axis=2)
            # actions = np.squeeze(actions)
            print("actions: ", actions.shape)
        if len(rewards.shape) >= 4:
            print("rewards: ", rewards.shape)
            rewards = np.squeeze(rewards, axis=2)
            # rewards = np.squeeze(rewards)
            print("rewards: ", rewards.shape)
        if len(lens.shape) >= 2:
            print("lens: ", lens.shape)
            lens = np.squeeze(lens, axis=1)
            # lens = np.squeeze(lens)
            print("lens: ", lens.shape)

        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).float()
        rewards = torch.from_numpy(rewards).float()
        lens = torch.from_numpy(lens).long()

    data = {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'lengths': lens
    }

    torch.save(data, args.pt_file)


if __name__ == '__main__':
    main()
