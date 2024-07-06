import argparse
import os
from distutils.util import strtobool

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp-name', type=str, default=os.path.basename(__file__).rstrip('.py'),
        help='the name of this experiment'
    )
    parser.add_argument(
        '--gym-id', type=str, default='ALE/Breakout-v5',
        help='the id of the gym environment'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=2.5e-4,
        help='optimiser learning rate'
    )
    parser.add_argument(
        '--seed', type=int, default=1,
        help='seed of the experiment'
    )
    parser.add_argument(
        '--total-timesteps', type=int, default=2_000_000,
        help='total timesteps of the experiments'
    )
    parser.add_argument(
        '--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`'
    )
    parser.add_argument(
        '--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will be enabled by default'
    )
    parser.add_argument(
        '--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
        help='whether to capture videos of the agent performances (check out `videos` folder)'
    )

    # Algorithm specific arguments
    parser.add_argument(
        '--num-envs', type=int, default=8, #128 in orig paper
        help='the number of parallel game environments'
    )
    parser.add_argument(
        '--num-steps', type=int, default=128,
        help='the number of steps to run in each environment per policy rollout'
    )
    parser.add_argument(
        '--anneal-lr', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Toggle learning rate annealing for policy and value networks'
    )
    parser.add_argument(
        '--gae', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Use GAE for advantage computation'
    )
    parser.add_argument(
        '--gamma', type=float, default=0.99,
        help='the discount factor gamma'
    )
    parser.add_argument(
        '--gae-lambda', type=float, default=0.95,
        help='the lambda for the general advantage estimation'
    )
    parser.add_argument(
        '--num-minibatches', type=int, default=4,
        help='the number of mini-batches'
    )
    parser.add_argument(
        '--update-epochs', type=int, default=4,
        help='the K epochs to update the policy'
    )
    parser.add_argument(
        '--norm-adv', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Toggles advantages normalization'
    )
    parser.add_argument(
        '--clip-coef', type=float, default=0.1,
        help='the surrogate clipping coefficient (epsilon)'
    )
    parser.add_argument(
        '--clip-vloss', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Toggles whether or not to use a clipped loss for the value function, as per the paper.'
    )
    parser.add_argument(
        '--ent-coef', type=float, default=0.01,
        help='coefficient of the entropy'
    )
    parser.add_argument(
        '--vf-coef', type=float, default=0.5,
        help='coefficient of the value function'
    )
    parser.add_argument(
        '--max-grad-norm', type=float, default=0.5,
        help='the maximum norm for the gradient clipping'
    )
    parser.add_argument(
        '--target-kl', type=float, default=None,
        help='the target KL divergence threshold'
    )
    parser.add_argument(
        '--ds-dim', type=float, default=84,
        help='the downsample dimension for the screen images'
    )

    # RND arguments
    parser.add_argument(
        '--update-portion', type=float, default=0.25,
        help='proportion of experience used for predictor update'
    )
    parser.add_argument(
        '--ext-coef', type=float, default=2.0,
        help='coefficient of extrinsic reward'
    )
    parser.add_argument(
        '--int-coef', type=float, default=1.0,
        help='coefficient of intrinsic reward'
    )
    parser.add_argument(
        '--int-gamma', type=float, default=0.99,
        help='Intrinsic reward discount rate'
    )
    parser.add_argument(
        '--n-init-obs-norm', type=float, default=0,
        help='number of iterations to initialise the observations normalisation parameters'
    )
    # parser.add_argument(
    #     '--pred-reset', type=int, default=5_000,
    #     help='Reset the predictor network final layer every n updates to network'
    # )
    # parser.add_argument(
    #     '--norm-momentum', type=float, default=0.1,
    #     help='Momentum to control the influence of new min and max values on running estimates in normalisation of behavioural loss tensors'
    # )
    parser.add_argument(
        '--dw-loss-coef', type=float, default=0.1,
        help='Used to rescale the Deathwatch masked-loss value when combining with overall loss'
    )
    parser.add_argument(
        '--combined-im-scalar', type=float, default=0.5,
        help='Used to rescale the combined reward bonus'
    )
    parser.add_argument(
        '--risk-threshold', type=float, default=0.6,
        help='Used to set the threshold below which risks are treated as acceptable (and get set to zero)'
    )


    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.squared_risk = args.risk_threshold ** 2
    # fmt: on
    return args
