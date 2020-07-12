import os

import gym

import numpy as np
import torch
from gym.spaces.box import Box
# from gym.wrappers import FlattenDictWrapper

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass


# class RelativeGoalFlattenDictWrapper(FlattenDictWrapper):
class RelativeGoalFlattenDictWrapper(gym.ObservationWrapper):
    def __init__(self, env, dict_keys):
        super(RelativeGoalFlattenDictWrapper, self).__init__(env)
        self.dict_keys = dict_keys

        size = 0
        for key in dict_keys:
            shape = self.env.observation_space.spaces[key].shape
            size += np.prod(shape)

        if all(item in dict_keys for item in ['desired_goal', 'achieved_goal']):
            size -= np.prod(self.env.observation_space.spaces['desired_goal'].shape)

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype='float32')

    def observation(self, observation):
        assert isinstance(observation, dict)
        obs = []

        obs.append(observation['observation'].ravel())
        obs.append((observation['desired_goal'] - observation['achieved_goal']).ravel())

        return np.concatenate(obs)


def make_env(args, env_id, seed, rank, log_dir, allow_early_resets):
    def _thunk():
        env_id = 'FurnitureBaxterEnv'
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        elif env_id.startswith("Fetch"):
            env = gym.make(env_id)
            env.env.reward_type = "dense"
            env = RelativeGoalFlattenDictWrapper(env, dict_keys=['observation', 'achieved_goal', 'desired_goal'])
        elif env_id.startswith("furniture"):
            """
            Returns argument parser for furniture assembly environment.
            """
            print("env_id: ", env_id)
            import argparse
            from furniture_tasks.util import str2bool

            import sys
            # sys.path.append('/Users/slee01/PycharmProjects/multi-task/furniture_tasks')

            furniture_parser = argparse.ArgumentParser("Demo for IKEA Furniture Assembly Environment")
            furniture_parser.add_argument('--env_id', type=str, default='furniture-baxter-v0')
            furniture_parser.add_argument('--num_env', type=int, default=1)
            furniture_parser.add_argument('--seed', type=int, default=123)
            furniture_parser.add_argument('--debug', type=str2bool, default=False)

            import furniture_tasks.config.furniture as furniture_config
            furniture_config.add_argument(furniture_parser)

            print("add furniture_parser")
            furniture_parser.set_defaults(visual_ob=True)
            print("/// after furniture_parser.set_defaults(visual_ob=True)")
            furniture_args = furniture_parser.parse_args()
            print("=== after furniture_args = furniture_parser.parse_args()")

            # for env in gym.envs.registry.env_specs:
            #     print(gym.envs.registry.env_specs[env])
            #     if 'furniture' in env:
            #         print('Remove {} from registr'.format(env))
            #         del gym.envs.registry.env_specs[env]

            # del gym.envs.registry.env_specs[env_id]
            env = gym.make(env_id, **furniture_args.__dict__)
            print("after gym.make")
        elif env_id.startswith("carla"):
            params = {
                'number_of_vehicles': args.carla_number_of_vehicles,
                'number_of_walkers': args.carla_number_of_walkers,
                'display_size': args.carla_display_size,  # screen size of bird-eye render
                'max_past_step': args.carla_max_past_step,  # the number of past steps to draw
                'dt': args.carla_dt,  # time interval between two frames
                'discrete': args.carla_discrete,  # whether to use discrete control space
                'ego_vehicle_filter': args.carla_ego_vehicle_filter,  # filter for defining ego vehicle
                'port': args.carla_port,  # connection port
                'town': args.carla_town,  # which town to simulate
                'task_mode': args.carla_task_mode,  # mode of the task, [random, roundabout (only for Town03)]
                'max_time_episode': args.carla_max_time_episode,  # maximum timesteps per episode
                'max_waypt': args.carla_max_waypt,  # maximum number of waypoints
                'obs_range': args.carla_obs_range,  # observation range (meter)
                'lidar_bin': args.carla_lidar_bin,  # bin size of lidar sensor (meter)
                'd_behind': args.carla_d_behind,  # distance behind the ego vehicle (meter)
                'out_lane_thres': args.carla_out_lane_thres,  # threshold for out of lane
                'desired_speed': args.carla_desired_speed,  # desired speed (m/s)
                'max_ego_spawn_times': args.carla_max_ego_spawn_times,  # maximum times to spawn ego vehicle
            }
            env = gym.make(env_id, params=params)
        else:
            env = gym.make(env_id)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)

        env.seed(seed + rank)

        obs_shape = env.observation_space.shape

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)

        if is_atari:
            if len(env.observation_space.shape) == 3:
                env = wrap_deepmind(env)
        elif len(env.observation_space.shape) == 3:
            raise NotImplementedError(
                "CNN models work only for atari,\n"
                "please use a custom wrapper for a custom pixel input env.\n"
                "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(args,
                  env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  num_frame_stack=None):
    envs = [
        make_env(args, env_name, seed, i, log_dir, allow_early_resets)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            if env_name.startswith("Fetch"):
                envs = VecNormalize(envs, ret=False, gamma=gamma)
            else:
                envs = VecNormalize(envs, gamma=gamma, ob=False)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, f"Error: Operation, {str(op)}, must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
