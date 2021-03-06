from gym.envs.registration import register


for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

    register(
        id='BaxterReach{}-v0'.format(suffix),
        entry_point='gym_baxter.envs:BaxterReachEnv',
        max_episode_steps=200,
        kwargs=kwargs,
    )
