from gym.envs.registration import register

register(
    id='skyjo-v0',
    entry_point='skyjo_gym_game.envs.skyjo_env:SkyjoEnv',
    max_episode_steps=2000
)