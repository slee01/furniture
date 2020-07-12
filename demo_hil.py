"""
Demonstration for RL experiments with new environment design.

This script tells you how to use our IKEA furniture assembly environment for RL
experiments and design your own reward function and task.

First, FurnitureExampleEnv shows you how to define a new task.
* `__init__`: sets environment- and task-specific configurations
* `_reset`: initializes variables when an episode is reset
* `_place_objects`: specifies the initialization of furniture parts
* `_get_obs`: includes more information for your task
* `_step`: simulates the environment and returns observation and reward
* `_compute_reward`: designs your own reward function

We describe how to collect trajectories with a random policy in `main`.

Please refer to `furniture/rl` for more advanced RL implementations.
"""


from collections import OrderedDict

import numpy as np

from env.furniture_baxter import FurnitureBaxterEnv
import env.transform_utils as T


def main(args):
    """
    Shows basic rollout code for simulating the environment.
    """
    print("IKEA Furniture Assembly Environment!")

    # make environment following arguments
    from env import make_env
    env = make_env('FurnitureBaxterEnv', args)

    # define a random policy
    def policy_action(ob):
        return env.action_space.sample()

    # define policy update
    def update_policy(rollout):
        pass

    # run one episode and collect transitions
    rollout = []
    done = False
    observation = env.reset()
    ep_length = 0

    # update unity rendering
    env.render()

    while not done:
        ep_length += 1

        # sample action from policy
        action = policy_action(observation)

        # simulate environment
        observation, reward, done, info = env.step(action)

        print('{:3d} step:  reward ({:5.3f})  action ({})'.format(
            ep_length, reward, action[:3]))

        # update unity rendering
        env.render()

        # collect transition
        rollout.append({'ob': observation,
                        'reward': reward,
                        'done': done})

    # update your network using @rollout
    update_policy(rollout)

    # close the environment instance
    env.close()


def argsparser():
    """
    Returns argument parser for furniture assembly environment.
    """
    import argparse
    import config.furniture as furniture_config
    from util import str2bool

    parser = argparse.ArgumentParser("Demo for IKEA Furniture Assembly Environment")
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--debug', type=str2bool, default=False)

    furniture_config.add_argument(parser)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argsparser()
    main(args)

