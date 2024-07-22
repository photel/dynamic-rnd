import random
import time
from collections import deque

# import gymnasium
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo, ResizeObservation, GrayScaleObservation, FrameStack
from gymnasium.wrappers.normalize import RunningMeanStd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (  
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from arg_parser_rnd import parse_args
from helpers import hardware, create_subdirectories, start_video


def make_env(env_id, seed=1, idx=0, capture_video=True, run_name='Unnamed_run'):
    '''
        Returns a function that creates a gym environment. This is convenient in order to use SyncVectorEnv, which takes a list of functions.
    '''
    def init_env():
        # Declare environment
        env = gym.make(env_id, render_mode='rgb_array')
        # Wrappers for monitoring progress
        env = RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = RecordVideo(env, video_folder=f'videos/{run_name}', episode_trigger=None, step_trigger=start_video, video_length=0)

        # SB3 preprocessors
        # Adds stochasticity by sampling a random number of no-ops between 0 and noop_max on env reset and returns resulting observation.
        # See Machado et al, 2017 https://arxiv.org/abs/1709.06009
        env = NoopResetEnv(env, noop_max=30)
        # Skips last x frames and repeats agent's last action on those frames. Saves computational time - Mnih et al, 2016 https://doi.org/10.1038/nature14236 p7
        env = MaxAndSkipEnv(env, skip=4)
        # End episode if all lives lost - Mnih et al, 2016
        env = EpisodicLifeEnv(env)
        # Triggers fire on envs that need it to start. Action can be learned so not strictly necessary.
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        # Clips the sum of rewards from the skipped frames to range (-1, +1)
        env = ClipRewardEnv(env)
        
        # Gymnasium Image transforms
        downsample_dim = (args.ds_dim, args.ds_dim)
        env = ResizeObservation(env, downsample_dim)
        env = GrayScaleObservation(env)
        # Stack 4 frames as a single obs to encode temporal information
        env = FrameStack(env, 4)
        
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return init_env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    '''
        - Initialise layer weights using orthogonal initialisation strategy for weights and a constant bias according to PPO requirements. 
        - .orthogonal_: Fills the input Tensor with a (semi) orthogonal matrix. Helps to avoid vanishing and exploding gradients (see: https://smerity.com/articles/2016/orthogonal_init.html).
        - .constant_: Initialises the specified network weights with the provided value
        
        Args:
            - layer: Input tensor
            - std: gain – optional scaling factor (affects standard deviation)
                - Important to use a gain that counteracts squeezing effect of Tanh activation (see: https://medium.com/@psharma_11665/initializing-weights-6ab48d2d99f0)
            - bias_const: value to fill constant_ tensor
    '''
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    '''
        PPO Agent with actor-critic architecture. 
        Uses twin value-head for estimating 
        extrinsic and intrinsic reward values.
    '''
    def __init__(self, envs):
        super(Agent, self).__init__()

        self.value_embedding = None

        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 448)),
            nn.ReLU(),
        )

        self.extra_layer = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.1), 
            nn.ReLU()
        )
        
        # Actor provides the policy network, outputs a distribution of unnormalised action probabilities.
        self.actor = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(448, envs.single_action_space.n), std=0.01),
        )

        # Critic evaluates the "goodness" of the current state. Uses twin value head for extrinsic and intrinsic rewards.
        self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)


    def get_value_embedding(self, x):
        '''
            Generate an embedding of the given state
            Args:
                - x: Observation
        '''
        # Rescale pixel values to (0,1)
        input_dims = x / 255.0
        # Features extracted by CNN with fully-connected layers
        cnn_fc = self.cnn(input_dims)
        embedding = self.extra_layer(cnn_fc)
        self.value_embedding = embedding

        return embedding, cnn_fc
    

    def get_value(self, x):
        '''
            Return value of current state.
            Args:
                - x: Observation
        '''
        embedding, cnn_fc = self.get_value_embedding(x)
        value_ext = self.critic_ext(embedding + cnn_fc)
        value_int = self.critic_int(embedding + cnn_fc)

        return value_ext, value_int


    def get_action_and_value(self, x, action=None):
        '''
            Run a forward pass through the actor and critic networks.
            Args:
                - x: Observation
                - action: Selected action
            Returns:
                - action: next action
                - action_log_prob: log_prob of taking next action
                - entropy: entropy regularisation term for the action probability distribution - can encourage exploration by reducing determinism in the policy.
                - value_ext: Extrinsic value of current state
                - value_int: Intrinsic value of current state
        '''
        # Generate feature maps and downstream embedding
        embedding, cnn_fc = self.get_value_embedding(x)
        logits = self.actor(cnn_fc)
        # Softmax to generate normalised action prob distribution
        probs = Categorical(logits=logits)

        if action is None:
            # During the rollout phase next action will be sampled
            action = probs.sample()

        action_log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        value_ext = self.critic_ext(embedding + cnn_fc)
        value_int = self.critic_int(embedding + cnn_fc)

        return action, action_log_prob, entropy, value_ext, value_int



class RNDModel(nn.Module):
    '''
        Calculates an intrinsic reward bonus based on encountering novel states.

        Target and predictor network each take state s(t+1) as input.
        The target network is not trained and the predictor network learns to minimise the error between their outputs. Intrinsic reward is computed on basis of the prediction error.
    '''
    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        feature_output = 7 * 7 * 64

        # Prediction network
        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        )

        # Target network
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
        )

        # target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature


class RewardForwardFilter:
    '''
        Compute cumulative returns.
    '''
    # See Curiosity PPO: https://github.com/openai/large-scale-curiosity/blob/0c3d179fd61ee46233199d0891c40fbe7964d3aa/cppo_agent.py#L226-L236
    def __init__(self, gamma):
        '''
            Arguments:
                - gamma: discount factor
        '''
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        '''
            Returns updated reward based on sum of new reward and discounted previous rewards.
            Arguments:
                - rews: Raw reward at a time step
        '''
        if self.rewems is None:
            self.rewems = rews
        else:
            # Update to new reward + discounted rewards
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems
    
# Intrinsic behaviour modules ----------

class DynamicSurprise(nn.Module):
    '''
        Takes an embedding from the Critic and appends the current action to generate input (s_rep, a).
        Predicts the output of the RND target network - i.e. a stable embedding of (st+1).
        Prediction error incorporates environment dynamics and can be considered as similar to surprise.
    '''
    def __init__(self, input_size=448, action_size=1):
        super(DynamicSurprise, self).__init__()
        
        # match output of net to output of RND target
        self.dy_sur_net = nn.Sequential(
            layer_init(nn.Linear(input_size + action_size, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512))
        )

    def forward(self, value_embedding, action):
        '''
            Arguments:
                - value_embedding: Embedding taken from Critic, prior to output head.
                - action: current action, selected by policy.
        '''
        # Ensure action is right shape & append to value_embedding
        unmatched_dim = 2
        action = action.unsqueeze(unmatched_dim)
        # concat the tensors which must match in all but excepted dim
        x = torch.cat((value_embedding, action), dim=unmatched_dim)
        # Forward pass
        x = self.dy_sur_net(x)
        return x


def get_room_number(ram):
    '''
        Get the Montezuma's Revenge room number from Atari RAM
        This is stored at RAM index 3 (https://arxiv.org/pdf/1611.04717)
    '''
    return ram[3]


# RUN -------    
if __name__ == '__main__':
    args = parse_args()
    print(args)

    # Name tensorboard project
    run_name = f'{args.gym_id}_{args.exp_name}_anticipate_{args.seed}_{int(time.time())}'
    # Save run data to folder
    writer = SummaryWriter(f'runs/{run_name}')
    # Encode args as text data
    writer.add_text(
        'hyperparameters',
        '|param|value|\n|-|-|\n%s' % ('\n'.join([f'|{key}|{value}|' for key, value in vars(args).items()])),
    )

    # Seeding so comparisons can be made across runs
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Use GPU if available
    device = hardware()

    # env setup ------------
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id=args.gym_id, seed=args.seed + i, idx=i, capture_video=args.capture_video, run_name=run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), 'only Discrete action space is supported'
    #  end env setup --------
    

    agent = Agent(envs).to(device)
    rnd_model = RNDModel(4, envs.single_action_space.n).to(device)
    dys_watch = DynamicSurprise().to(device)
    
    # store the concatenated list of parameters from the agent and RND predictor and additional behaviour modules
    combined_parameters = list(agent.parameters()) + list(rnd_model.predictor.parameters()) + list(dys_watch.parameters())

    # note: epsilon is added to denominator for numerical stability
    optimiser = optim.Adam(
        combined_parameters,
        lr=args.learning_rate,
        eps=1e-5
    )

    # Setup utilities to get the running mean and standard deviation statistics as new experience received.
    # Used for reward normalisation, see Burda et al 2018, p.4 - https://doi.org/10.48550/arXiv.1808.04355
    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, args.ds_dim, args.ds_dim))
    discounted_reward = RewardForwardFilter(args.int_gamma)


    rollout_size = (args.num_steps, args.num_envs)
    # Setup storage of experience rollout data
    obs = torch.zeros(rollout_size + envs.single_observation_space.shape).to(device)
    actions = torch.zeros(rollout_size + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros(rollout_size).to(device)
    rewards = torch.zeros(rollout_size).to(device)
    # Intrinsic behaviour bonuses -------
    intrinsic_rewards = torch.zeros(rollout_size).to(device)
    value_embed_shape = torch.Size([8, 448])
    value_embeds = torch.zeros(rollout_size + value_embed_shape).to(device)
    dones = torch.zeros(rollout_size).to(device)
    ext_values = torch.zeros(rollout_size).to(device)
    int_values = torch.zeros(rollout_size).to(device)
    avg_returns = deque(maxlen=20) # May not need

    # Keep track of visited rooms
    room_visits = [set() for _ in range(args.num_envs)]
    env_episode_counts = torch.zeros(args.num_envs).to(device)


    # Track combined steps across all environments
    global_step = 0
    global_episode_count = 0
    start_time = time.time()
    
    # Init next_obs with first initial observation
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    # Store initial termination condition as False
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size


    print('Start initialising observation normalisation parameter...')
    # during this phase take random actions and store the observations to intialise obs_rms, which is used in normalising observations - this is needed to stabilise the RND error
    next_ob = []
    for step in range(args.num_steps * args.n_init_obs_norm):
        acs = np.random.randint(0, envs.single_action_space.n, size=(args.num_envs,))
        st, rw, te, tr, _ = envs.step(acs)
        next_ob += st[:, 3, :, :].reshape([-1, 1, args.ds_dim, args.ds_dim]).tolist()

        if len(next_ob) % (args.num_steps * args.num_envs) == 0:
            next_ob = np.stack(next_ob)
            # Update the running mean and std deviation with the new data
            obs_rms.update(next_ob)
            next_ob = []
    print('...end intialisation')


    # Network parameter updates
    for update in range(1, num_updates+1):
        # Learning rate annealing based on input flag
        if args.anneal_lr:
            # Start fraction at 1 and decrease linearly per update
            lr_frac = 1.0 - ((update - 1.0) / num_updates)
            # Set new learning rate value
            lr_new = lr_frac * args.learning_rate
            # Use PyTorch API to update learning rate
            optimiser.param_groups[0]['lr'] = lr_new

        # Policy rollout
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            # Store the obs and done data
            obs[step] = next_obs
            dones[step] = next_done

            # During rollout don't need to calculate the gradient so use torch.no_grad context
            with torch.no_grad():
                # Get the twin value predictions
                value_ext, value_int = agent.get_value(obs[step])
                ext_values[step], int_values[step] = (
                    value_ext.flatten(),
                    value_int.flatten(),
                )
                action, logprob, _, _, _ = agent.get_action_and_value(obs[step])
                
                # Store value embeddings during rollout
                value_embeds[step] = agent.value_embedding

            actions[step] = action
            logprobs[step] = logprob

            # Room tracking for MZR
            if 'MontezumaRevenge' in args.gym_id:
                for i in range(args.num_envs):
                    # ram = envs[i].unwrapped.ale.getRAM()
                    # ram = envs.env.ale.getRAM()
                    ram = envs.envs[i].env.ale.getRAM()
                    room_number = get_room_number(ram)
                    room_visits[i].add(room_number)



            # Stepping the environment
            # Note that we transfer the env from GPU to CPU to do the step
            next_obs, reward, term, trunc, info = envs.step(action.cpu().numpy())

            # Set done for an update if env terminates OR truncates
            done_signals = zip(term, trunc)
            done = [a or b for a, b in done_signals]

            # Tranfer rewards etc. to the GPU
            # .view(-1) flattens a tensor
            # specify float32 tensor for mps
            if device == 'mps':
                rewards[step] = torch.tensor(reward).to(device, dtype=torch.float32).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device, dtype=torch.float32), torch.Tensor(done).to(device, dtype=torch.float32)
            else:
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
                        
            # Reshape the observation
            next_obs_reshape = next_obs[:, 3, :, :].reshape(args.num_envs, 1, args.ds_dim, args.ds_dim)

            # Standardise the observation
            if device == 'mps':
                next_obs_std = (next_obs_reshape - torch.from_numpy(obs_rms.mean).to(device, dtype=torch.float32)) / torch.sqrt(torch.from_numpy(obs_rms.var).to(device, dtype=torch.float32))
            else:
                next_obs_std = (next_obs_reshape - torch.from_numpy(obs_rms.mean).to(device)) / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
            
            # Clip the observation to range [-5, 5] to stabilise computation in networks
            rnd_next_obs = ((next_obs_std).clip(-5, 5)).float()
            
            # Get the feature representation of the next state from the fixed target network
            target_next_feature = rnd_model.target(rnd_next_obs)
            # Get the predicted feature representation of the next state from the trainable network
            predict_next_feature = rnd_model.predictor(rnd_next_obs)

            # Env dynamics module predictions
            dy_sur_pred = dys_watch(agent.value_embedding.unsqueeze(0), action.unsqueeze(0)).squeeze()

            # Get the MSE to generate the reward bonus
            # rnd_pred_error = target_next_feature - predict_next_feature
            rnd_pred_error = target_next_feature - predict_next_feature
            dy_sur_pred_error = target_next_feature - dy_sur_pred

            rnd_pred_error = rnd_pred_error.cpu().detach().numpy()
            dy_sur_pred_error = dy_sur_pred_error.cpu().detach().numpy()

            # Generate a weighted ds error
            dy_sur_err_weighted = dy_sur_pred_error * args.dw_loss_coef
            combined_intrinsic_error = rnd_pred_error + dy_sur_err_weighted
            ie_tensor = torch.from_numpy(combined_intrinsic_error)
            intrinsic_rewards[step] = ((ie_tensor).pow(2).sum(1) / 2).data

            # ------ LOGGING ------
            # Check for any steps that returned done
            for idx, d in enumerate(done):
                # If end of episode log data to Tensorboard
                if d and info['lives'][idx] == 0:
                    for item in info:
                        if item == 'final_info':
                            for item_data in info[item]:
                                if item_data and 'episode' in item_data.keys():
                                    global_episode_count += 1
                                    writer.add_scalar('charts/episodic_return', item_data['episode']['r'], global_step)
                                    writer.add_scalar('charts/episodic_length', item_data['episode']['l'], global_step)
                                    writer.add_scalar('charts/episode_count', global_episode_count, global_step)

                                    if 'MontezumaRevenge' in args.gym_id:
                                        # Log rooms visited in an episode
                                        env_episode_counts[idx] += 1
                                        visit_count = len(room_visits[idx])
                                        room_visits_str = ', '.join(map(str, sorted(room_visits[idx])))
                                        writer.add_scalar('rooms/per_step', visit_count, global_step)
                                        writer.add_scalar('rooms/per_episode', visit_count, env_episode_counts[idx])

                                        # Log room visits to file
                                        room_subdirectories = [run_name]
                                        room_log_path = create_subdirectories('room_logs', room_subdirectories)
                                        room_logfile = f'./room_logs/{run_name}/rooms.log'
                                        with open(room_logfile, 'a') as f:
                                            f.write(f'Episode {env_episode_counts[idx]} (env {idx}): {room_visits_str}\n')
                    
                                    break
            
            # Save model - re-enable this for model saving
            # if (global_step > 0 and global_step % 1000_000 == 0) or (global_step == args.num_steps - 5000):
            #     subdirectories = [run_name, f'step_{global_step}']
            #     path_str = create_subdirectories('rnd_models', subdirectories)
            #     torch.save(agent.state_dict(), f'{path_str}/agent_model.pt')
            #     torch.save(rnd_model.predictor.state_dict(), f'{path_str}/rnd_model.pt')

        # Update the intrinsic bonuses
        intrinsic_reward_per_env = np.array(
            [discounted_reward.update(step_reward) for step_reward in intrinsic_rewards.cpu().data.numpy().T]
        )
        
        mean, std, count = (
            np.mean(intrinsic_reward_per_env),
            np.std(intrinsic_reward_per_env),
            len(intrinsic_reward_per_env),
        )
        # For the reward, update a running mean and std using the computed mean, the std^2 (equivalent to the variance), and the count of elements
        reward_rms.update_from_moments(mean, std**2, count)
        # Normalise rewards with square root of the running variance
        intrinsic_rewards /= np.sqrt(reward_rms.var)

        # Log curiosity reward stats
        writer.add_scalar('curiosity/reward_mean', mean, global_step)
        writer.add_scalar('curiosity/reward_std', std, global_step)
        writer.add_scalar('curiosity/reward_count', count, global_step)


        # Computing the Generalised Advantage Estimation (GAE)
        with torch.no_grad():
            # Estimate value of next state
            next_value_ext, next_value_int = agent.get_value(next_obs)
            next_value_ext, next_value_int = next_value_ext.reshape(1, -1), next_value_int.reshape(1, -1)
            # Setup storage for extrinsic and intrinsic advantage estimations
            ext_advantages = torch.zeros_like(rewards, device=device)
            int_advantages = torch.zeros_like(intrinsic_rewards, device=device)
            ext_lastgaelam = 0
            int_lastgaelam = 0
            
            for t in reversed(range(args.num_steps)):
                # update next values and nonterminal flags
                if t == args.num_steps - 1:
                    # from latest data
                    ext_nextnonterminal = 1.0 - next_done
                    int_nextnonterminal = 1.0
                    ext_nextvalues = next_value_ext
                    int_nextvalues = next_value_int
                else:
                    # from stored data
                    ext_nextnonterminal = 1.0 - dones[t + 1]
                    int_nextnonterminal = 1.0
                    ext_nextvalues = ext_values[t + 1]
                    int_nextvalues = int_values[t + 1]
                
                # temporal difference in extrinsic rewards
                # NOTE: ext_delta represents the temporal difference in extrinsic rewards. It is computed as the difference between the observed extrinsic reward at time step t (rewards[t]) and the estimated value of the extrinsic rewards at time step t (ext_values[t]), plus a discounted estimate of the extrinsic rewards at the next time step (args.gamma * ext_nextvalues * ext_nextnonterminal) -- also for int_delta
                ext_delta = rewards[t] + (args.gamma * ext_nextvalues * ext_nextnonterminal) - ext_values[t]
                # temporal difference in intrinsic rewards
                int_delta = intrinsic_rewards[t] + (args.int_gamma * int_nextvalues * int_nextnonterminal) - int_values[t]
                
                # get advantage for extrinsic reward - uses Generalized Advantage Estimation (GAE) with a lambda param
                # Advantage estimate quantifies how much better the observed outcome of choosing an action in a given state was versus the state’s value, estimated by the critic.
                ext_advantages[t] = ext_lastgaelam = (
                    ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                )
                # get advantage for intrinsic reward
                int_advantages[t] = int_lastgaelam = (
                    int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam
                )

            ext_returns = ext_advantages + ext_values
            int_returns = int_advantages + int_values


        # flatten the batch data
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions_basic = actions.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_ext_advantages = ext_advantages.reshape(-1)
        b_int_advantages = int_advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_ext_values = ext_values.reshape(-1)
        b_value_embeds = value_embeds.reshape((-1,) + (8, 448)) # Batched value embeddings

        b_advantages = b_int_advantages * args.int_coef + b_ext_advantages * args.ext_coef

        # Update observation running mean & std from batched observations
        obs_rms.update(b_obs[:, 3, :, :].reshape(-1, 1, args.ds_dim, args.ds_dim).cpu().numpy())

        # ---- Optimizing the policy and value network
        # For training we need all of the indices of the batch
        b_inds = np.arange(args.batch_size)

        # From the batch we need the standardised, clipped observation for the RND module
        b_obs_reshape = b_obs[:, 3, :, :].reshape(-1, 1, args.ds_dim, args.ds_dim)

        if device == 'mps':
            b_obs_std = (b_obs_reshape - torch.from_numpy(obs_rms.mean).to(device, dtype=torch.float32)) / torch.sqrt(torch.from_numpy(obs_rms.var).to(device, dtype=torch.float32))
        else:
            b_obs_std = (b_obs_reshape - torch.from_numpy(obs_rms.mean).to(device)) / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))

        rnd_next_obs = ((b_obs_std).clip(-5, 5)).float()
        clipfracs = [] # Use this for logging frequency of clipping
        for epoch in range(args.update_epochs):
            # Shuffle the batch indices for each training update epoch
            np.random.shuffle(b_inds)
            # Loop through each batch, a minibatch at a time
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # ---------- Apply RND model to a batch of observations and return features for both the predictor and target networks
                predict_next_state_feature, target_next_state_feature = rnd_model(rnd_next_obs[mb_inds])
                
                # Get MSE loss between the predictor and target features
                # .detach() target features from the computation graph, as these features should not be part of the backprop gradient update
                # reduction=none means that instead of computing a single scalar loss value for the entire batch MSE loss is calculated separately for each item in the batch, yielding a tensor of loss values. The .mean() across the elements of each observation in the batch is then computed, returning a scalar value for each.
                forward_loss = F.mse_loss(
                    predict_next_state_feature, target_next_state_feature.detach(), reduction="none"
                ).mean(-1)
                
                # Behaviour module predictions
                dyna_surp_pred = dys_watch(b_value_embeds[mb_inds], b_actions[mb_inds])
                # Get prediction loss
                target_next_state_feature_expanded = target_next_state_feature.unsqueeze(1).expand(-1, dyna_surp_pred.size(1), -1)

                dyna_surp_loss = F.mse_loss(
                    dyna_surp_pred, target_next_state_feature_expanded.detach(), reduction="none"
                ).mean(-1)

                dyna_surp_loss = dyna_surp_loss.reshape(-1)
                mask = torch.rand(len(forward_loss), device=device)
                dy_sur_mask = torch.rand(len(dyna_surp_loss), device=device)
                # Convert random mask to binary mask. Elements become 1 if < args.update_portion and 0 otherwise. Convert to FloatTensor and move to device. This lets us set the proportion of experience from the batch that gets used for the predictor update. This should provide some regularisation.
                mask = (mask < args.update_portion).type(torch.FloatTensor).to(device)
                dy_sur_mask = (dy_sur_mask < args.update_portion).type(torch.FloatTensor).to(device)
                # Sets some of the forward_loss values to 0, then gets the mean of the resulting values and returns the max of that mean or 1.
                forward_loss = (forward_loss * mask).sum() / torch.max(
                    mask.sum(), torch.tensor([1], device=device, dtype=torch.float32)
                )
                dyna_surp_loss = (dyna_surp_loss * dy_sur_mask).sum() / torch.max(
                    dy_sur_mask.sum(), torch.tensor([1], device=device, dtype=torch.float32)
                )
                
                # Carry out a forward pass on a minibatch of observations. We also pass in the minibatch actions so the agent won't sample new actions.
                _, newlogprob, entropy, new_ext_values, new_int_values = agent.get_action_and_value(b_obs[mb_inds], b_actions_basic.long()[mb_inds])
                # Logarithmic subtraction between the new logprobs and the old logprobs associated with the actions from the policy rollout phase. This provides the ratio for the surrogate objective clipping function.
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                # Some debug vars
                with torch.no_grad():
                    # --- KL-divergence (diff between distributions) helps us understand how agressively the policy updates
                    # PPO KL calculation
                    old_approx_kl = (-logratio).mean()
                    # Newer approach from Schulman http://joschu.net/blog/kl-approx.html
                    approx_kl = ((ratio - 1) - logratio).mean()
                    # --- Track how often the clipped objective is triggered
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                
                mb_advantages = b_advantages[mb_inds]
                # Apply PPO's advantage normalisation
                if args.norm_adv:
                    # Include small scalar 1e-8 to avoid divide by 0 error
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)


                # Setup PPO's clipped objective ---------
                # Clipping is core to PPO's stabilisation of training by avoiding too-large parameter updates

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                # Here we take max of negatives whereas PPO paper minimises a positive - these are equivalent!
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Adapting PPOs clipped Value loss
                new_ext_values = new_ext_values.view(-1)
                new_int_values = new_int_values.view(-1)
                # Use PPO's clipped loss if flag set
                if args.clip_vloss:
                    # Unclipped loss is the squared difference between the predicted values and the actual returns
                    ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
                    ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
                        new_ext_values - b_ext_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    # squared loss between the clipped predicted extrinsic values (ext_v_clipped) and the corresponding extrinsic returns (b_ext_returns) for a minibatch of examples (mb_inds)
                    ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_inds]) ** 2
                    # Get max of unclipped and clipped loss
                    ext_v_loss_max = torch.max(ext_v_loss_unclipped, ext_v_loss_clipped)
                    # Final value loss (ext_v_loss) is half of the mean of the max losses. 0.5 is commonly used to add numerical stability
                    ext_v_loss = 0.5 * ext_v_loss_max.mean()
                else:
                    # A more vanilla value loss uses the MSE between the predicted values and the actual returns
                    ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()

                # No clipping applies on intrinsic value loss - MSE between the predicted values and the actual returns
                int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                # Total loss is sum of ext and int losses
                v_loss = ext_v_loss + int_v_loss

                # Entropy here measures the disorder in the action probability distribution - maximising entropy encourages exploration
                entropy_loss = entropy.mean()
                # Final loss is a sum of policy loss, weighted value loss, forward loss between the RND predictor and target nets, less the weighted entropy loss combined with the dyna_surp_loss.
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + forward_loss + dyna_surp_loss

                # zero out the gradients before each optimization step to avoid them being accumulated across multiple batches
                optimiser.zero_grad()
                # Perform backprop
                loss.backward()
                # If max norm threshold is set use it for clipping to prevent exploding gradients -- Maximum norm gradient clipping calculates the norm (magnitude) of the entire gradient vector and scales it down proportionally if it exceeds the supplied threshold
                if args.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        combined_parameters,
                        args.max_grad_norm,
                    )
                # Update the network weights
                optimiser.step()


            # Early stopping. Try setting target-kl=0.015 if using - as per 'OpenAI Spinning Up' - Using it at batch level here but can apply at minibatch instead for the craic!
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break


        # Record metrics for Tensorboard
        writer.add_scalar('charts/learning_rate', optimiser.param_groups[0]['lr'], global_step)
        writer.add_scalar('losses/value_loss', v_loss.item(), global_step)
        writer.add_scalar('losses/policy_loss', pg_loss.item(), global_step)
        writer.add_scalar('losses/entropy', entropy_loss.item(), global_step)
        writer.add_scalar('losses/old_approx_kl', old_approx_kl.item(), global_step)
        writer.add_scalar('losses/fwd_loss', forward_loss.item(), global_step)
        writer.add_scalar('losses/dyna_surp_loss', dyna_surp_loss.item(), global_step)
        writer.add_scalar('losses/approx_kl', approx_kl.item(), global_step)
        writer.add_scalar('losses/clipfrac', np.mean(clipfracs), global_step)
        print('Steps per second:', int(global_step / (time.time() - start_time)))
        writer.add_scalar('charts/SPS', int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
