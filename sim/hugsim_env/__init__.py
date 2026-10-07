from gymnasium.envs.registration import register


register(
     id="hugsim_env/HUGSim-v0",
     entry_point="hugsim_env.envs:HUGSimEnv",
     # No step-count TimeLimit: 400 steps is 100 s at the shipped dt = 0.25 s but only 40 s at
     # dt = 0.1, so a route that needs 60 s became impossible to finish purely by running the
     # simulator faster. The episode length is a duration and closed_loop.py owns it
     # (EPISODE_SECONDS), which cuts at the same step as this wrapper did at the shipped dt.
     max_episode_steps=None,
)