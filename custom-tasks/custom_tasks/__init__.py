from gym.envs.registration import register

register(
    id='MountainToyCar-v1',
    entry_point='custom_tasks.envs:MountainToyCarEnv',
    max_episode_steps=400,
    reward_threshold=-110.0,
)
register(
    id='MountainToyCarContinuous-v1',
    entry_point='custom_tasks.envs:MountainToyCarContinuousEnv',
    max_episode_steps=400,
    reward_threshold=-110.0,
)

register(
    id='carla-v1',
    entry_point='custom_tasks.envs:CarlaEnv',
    max_episode_steps=10000,
    reward_threshold=-110.0,
)