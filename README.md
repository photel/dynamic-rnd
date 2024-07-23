# Dynamic-RND
This codebase presents _Dynamic-RND (D-RND)_, a deep reinforcement learning model that aims to improve exploration via an intrinsic reward signal. The model incorporates the curiosity-driven approach of Random Network Distillation (RND) [(Burda et al, 2018)](https://arxiv.org/abs/1810.12894) with a Proximal Policy Optimsation (PPO) [(Schulman et al, 2017)](https://arxiv.org/abs/1707.06347) agent. It builds upon a CleanRL [(Huang et al, 2022)](https://github.com/vwxyzjn/cleanrl) implementation of those models, which is adapted to work with [Gymnasium](https://gymnasium.farama.org/index.html) and to optionally train on Apple's Metal architecture.

The model has been developed as partial fulfilment of a part-time MSc in artificial intelligence at University of Limerick.

The model is setup to run on the Atari Arcade Learning Environment.

## Setup local environment
Dependencies are included in the environment.yml file. Ensure you have Conda installed and run:
`conda env create -f environment.yml`.
To activate the environment run: `conda activate Pytorch-P310`.
It may also be necessary to install the following manually as they can be problematic/impossible to install from the environment yaml file.

- `pip install swig`
- `pip install 'gymnasium[box2d]'`
- `pip install 'gymnasium[atari, accept-rom-license]'`

## Training the models
### D-RND
Parameters for running the D-RND model are included in the **arg_parser_rnd.py** file. These can be passed via the command line when the model is being trained.

A sample run command for the D-RND model is as follows:

`python dynamic_rnd.py --gym-id ALE/MontezumaRevenge-v5 --total-timesteps 50000000 --seed 1`

---

There are working adaptations of PPO and RND models that have been used for establishing baselines for the purposes of evaluating the model. Instructions for running these are included below.

### PPO
Parameters for running the PPO model are included in the **arg_parser.py** file. These can be passed via the command line when the model is being trained.

A sample run command for the PPO model is as follows:

`python ppo_atari.py --gym-id ALE/MontezumaRevenge-v5 --total-timesteps 50000000 --seed 1`

### RND
Parameters for running the RND model are included in the **arg_parser_rnd.py** file. These can be passed via the command line when the model is being trained.

A sample run command for the RND model (using PPO) is as follows:

`python rnd_ppo.py --gym-id ALE/MontezumaRevenge-v5 --total-timesteps 50000000 --seed 1`

## Logs
To view Tensorboard logs for experiments run `tensorboard --logdir runs`, where _**runs**_ is the name of the directory containing the logs.
