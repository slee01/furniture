import argparse
import os
import sys
import pickle
import glob
import copy

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


def convertObject2TorchArray(featuresArray, target_size):
    for features in featuresArray:
        _pad = copy.deepcopy(features[-1])

        while len(features) < target_size:
            features.append(_pad)

    featuresArray = torch.from_numpy(np.array(featuresArray))
    return featuresArray


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

    if args.pt_file is None:
        args.pt_file = os.path.splitext(args.pkl_file)[0] + '.pt'

    assert (
            args.pkl_file is not None
    ), "--demo_path should be set (e.g. demos/Sawyer_toy_table)"
    demo_files = get_demo_files(args.pkl_file)

    print("file_path: ", args.pkl_file)
    print("pt_file: ", args.pt_file)
    print("demo_files: ", demo_files)

    statesArray, actionsArray, rewardsArray, lenArray = [], [], [], []

    # now load the picked numpy arrays
    for file_path in demo_files:
        states, actions, rewards = [], [], []

        with open(file_path, "rb") as f:
            demo = pickle.load(f)
            # print("demo: ", type(demo), demo.keys())

            # add observations
            for state in demo["obs"]:
                state = np.concatenate(list(state.values()))
                states.append(state)
            states.pop()

            # add actions
            for action in demo["actions"]:
                actions.append(action)

            # add rewards
            if "rewards" in demo:
                for reward in demo["rewards"]:
                        rewards.append(reward)

            # dataset_size = len(states)

            # print("data: ", file_path)
            # print("states: ", np.array(states).shape)
            # print("actions: ", np.array(actions).shape)
            # print("rewards: ", np.array(rewards).shape)

        statesArray.append(states)
        actionsArray.append(actions)
        rewardsArray.append(rewards)
        lenArray.append(len(states))

    statesArray = convertObject2TorchArray(statesArray, np.max(lenArray)).float()
    actionsArray = convertObject2TorchArray(actionsArray, np.max(lenArray)).float()
    rewardsArray = convertObject2TorchArray(rewardsArray, np.max(lenArray)).reshape(-1, np.max(lenArray), 1).float()
    lenArray = torch.from_numpy(np.array(lenArray)).long()
    print("states: ", statesArray.shape)
    print("actions: ", actionsArray.shape)
    print("rewards: ", rewardsArray.shape)
    print("lens: ", lenArray.shape)

    # (trajs, episodes, features)
    # states: torch.Size([3000, 50, 28])
    # actions: torch.Size([3000, 50, 4])
    # rewards: torch.Size([3000, 50, 1])
    # lens: torch.Size([3000])

    data = {
        'states': statesArray,
        'actions': actionsArray,
        'rewards': rewardsArray,
        'lengths': lenArray
    }

    torch.save(data, args.pt_file)


if __name__ == '__main__':
    main()
