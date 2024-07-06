# Proximal Policy Optimisation (PPO)

- This implementation of PPO is based on excellent work by [Costa Huang et al.](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).
- It adapts their implementation to work with Farama's Gymnasium environments.
- See arg_parser.py for arguments that can be passed via the command line.

A sample run command for a basic env is as follows:

`python ppo.py --gym-id CartPole-v1 --total-timesteps 100000 --capture-video`

A sample run command for an Atari env is as follows:

`python ppo_atari.py --gym-id PongNoFrameskip-v4 --total-timesteps 2000000 --capture-video True`

A sample run command for the Random Network Distillation model (using PPO) is as follows:

`python rnd_ppo.py --gym-id ALE/MontezumaRevenge-v5 --total-timesteps 2000000 --capture-video True`

## Logs
To view Tensorboard logs for experiments run `tensorboard --logdir runs`, where _**runs**_ is the name of the directory containing the logs.
