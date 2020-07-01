""" Define all environments and provide helper functions to load environments. """

from env.base import make_env, make_vec_env

# register all environment to use
from env.furniture_baxter import FurnitureBaxterEnv
from env.furniture_sawyer import FurnitureSawyerEnv
from env.furniture_cursor import FurnitureCursorEnv
from env.furniture_baxter_block import FurnitureBaxterBlockEnv

import gym
# OpenAI gym interface
from gym.envs.registration import register

env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'furniture' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]

# add cursor environment to Gym
register(
    id='furniture-cursor-v0',
    entry_point='env.furniture_gym:FurnitureGym',
    kwargs={'name': 'FurnitureCursorEnv',
            'furniture_id': 0,
            'background': 'Lab',
            'port': 1050}
)


# add sawyer environment to Gym
register(
    id='furniture-sawyer-v0',
    entry_point='env.furniture_gym:FurnitureGym',
    kwargs={'name': 'FurnitureSawyerEnv',
            'furniture_name': 'swivel_chair_0700',
            'background': 'Industrial',
            'port': 1050}
)


# add baxter environment to Gym
register(
    id='furniture-baxter-v0',
    entry_point='env.furniture_gym:FurnitureGym',
    kwargs={'name': 'FurnitureBaxterEnv',
            'furniture_id': 1,
            'background': 'Interior',
            'port': 1050}
)

