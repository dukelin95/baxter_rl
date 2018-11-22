from gym.envs.registration import register

register(
    id='baxter-reach-v0',
    entry_point='gym_baxter.envs:baxter_fetch'
    )
