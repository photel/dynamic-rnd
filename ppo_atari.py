import random
import time

# import gymnasium
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo, ResizeObservation, GrayScaleObservation, FrameStack
# from gymnasium.experimental.vector import SyncVectorEnv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from arg_parser import parse_args
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
            # if idx == 0:
            env = RecordVideo(env, video_folder=f'videos/{run_name}', episode_trigger=None, step_trigger=start_video, video_length=5000)

        # SB3 preprocessors
        # Adds stochasticity by sampling a random number of no-ops between 0 and noop_max on env reset and returns resulting observation.
        # See Machado et al, 2017 https://arxiv.org/abs/1709.06009
        env = NoopResetEnv(env, noop_max=30)
        # Skips last x frames and repeats agent's last action on those frames. Saves computational time - Mnih et al, 2016 https://doi.org/10.1038/nature14236
        env = MaxAndSkipEnv(env, skip=4)
        # End episode if all lives lost - Mnih et al, 2016
        env = EpisodicLifeEnv(env)
        # Triggers fire on envs that need it to start. Action can be learned so not strictly necessary.
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        # Clips the sum of rewards from the skipped frames to range (-1, +1)
        env = ClipRewardEnv(env)
        
        # Gymnasium Image transforms
        env = ResizeObservation(env, (84, 84))
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
        - .orthogonal_: Fills the input Tensor with a (semi) orthogonal matrix
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
    def __init__(self, envs):
        super(Agent, self).__init__()

        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )

        # Actor provides the policy network, outputs a distribution of unnormalised action probabilities.
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        # Critic evaluates the "goodness" of the current state.
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        '''

            Return value of current state.
            Args:
                - x: Observation
        '''
        # Rescale pixel values to (0,1)
        input_dims = x / 255.0
        return self.critic(self.cnn(input_dims))


    def get_action_and_value(self, x, action=None):
        '''
            Run a forward pass through the actor and critic networks.
            Args:
                - x: Observation
                - action: Selected action
            Returns:
                - action: next action
                - action_log_prob: log_prob of taking next action
                - entropy: entropy regularisation term for the action probability distribution
                - value: Value of current state
        '''
        # Rescale pixel values to (0,1)
        input_dims = x / 255.0
        cnn_flat = self.cnn(input_dims)
        logits = self.actor(cnn_flat)
        # Softmax to generate normalised action prob distribution
        probs = Categorical(logits=logits)

        if action is None:
            # During the rollout phase next action will be sampled
            action = probs.sample()

        action_log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        value = self.critic(cnn_flat)

        return action, action_log_prob, entropy, value


# RUN -------    

if __name__ == '__main__':
    args = parse_args()
    print(args)

    # Name tensorboard project
    run_name = f'{args.gym_id}_{args.exp_name}_{args.seed}_{int(time.time())}'
    # Save run data to folder
    writer = SummaryWriter(f'runs/{run_name}')
    # Encode args as text data
    writer.add_text(
        'hyperparameters',
        '|param|value|\n|-|-|\n%s' % ('\n'.join([f'|{key}|{value}|' for key, value in vars(args).items()])),
    )
    # some dummy test data for Tensorboard
    # for i in range(1000):
    #     writer.add_scalar('test_loss', i*2, global_step=i)

    # To run tensorboard, in proj directory: tensorboard --logdir runs

    # Don't modify seeding so comparisons can be made across runs
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
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    #  end env setup --------


    agent = Agent(envs).to(device)
    # epsilon is added to denomintor to prevent 0-division errors
    optimiser = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Setup storage of experience rollout data
    rollout_size = (args.num_steps, args.num_envs)
    obs = torch.zeros(rollout_size + envs.single_observation_space.shape).to(device)
    actions = torch.zeros(rollout_size + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros(rollout_size).to(device)
    rewards = torch.zeros(rollout_size).to(device)
    dones = torch.zeros(rollout_size).to(device)
    values = torch.zeros(rollout_size).to(device)


    # Track combined steps across all environments
    global_step = 0
    global_episode_count = 0
    global_epochs = 0
    start_time = time.time()
    # Init next_obs with first initial observation
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    # Store initial termination condition as False
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    

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

            # During rollout don't need to cache the gradient so use torch.no_grad context
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                # Store the returned data
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Stepping the environment
            # Note that we transfer the env from GPU to CPU to do the step
            next_obs, reward, term, trunc, info = envs.step(action.cpu().numpy())
            # Set done for an update if env terminates OR truncates
            done = [a or b for a, b in zip(term, trunc)]

            # Tranfer the returned rewards to the GPU
            # specify dtype=torch.float32 if using MPS chipset
            if device == 'mps':
                rewards[step] = torch.tensor(reward).to(device, dtype=torch.float32).view(-1)
            else:
                rewards[step] = torch.tensor(reward).to(device).view(-1)

            # Transfer next_obs & next_done to GPU
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            
            # Log data to Tensorboard
            for item in info:
                if item == 'final_info':
                    for item_data in info[item]:
                        if item_data and 'episode' in item_data.keys():
                            global_episode_count += 1
                            writer.add_scalar('charts/episodic_return', item_data['episode']['r'], global_step)
                            writer.add_scalar('charts/episodic_length', item_data['episode']['l'], global_step)
                            writer.add_scalar('charts/episode_count', global_episode_count, global_step)
                            break
            
            # Save model
            if global_step > 0 and global_step % 1000_000 == 0 or (global_step == args.num_steps - 5000):
                subdirectories = [run_name, f'step_{global_step}']
                path_str = create_subdirectories('ppo_models', subdirectories)
                torch.save(agent.state_dict(), f'{path_str}/model.pt')

        # PPO bootstraps values if the environment is not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)

            if args.gae:
                # Use General Advantage Estimation (if flagged) based on PPO paper
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                # Use ordinary advantage estimation
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values
        
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        # For training we need all of the indices of the batch
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            global_epochs += 1
            # Shuffle the batch indices for each training update epoch
            np.random.shuffle(b_inds)
            # Loop through each batch, a minibatch at a time
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Carry out a forward pass on a minibatch of observations. We also pass in the minibatch actions so the agent won't sample new actions.
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                # Logarithmic subtraction between the new logprobs and the old logprobs associated with the actions from the policy rollout phase.
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Some debug vars
                with torch.no_grad():
                    # --- KL-divergencs helps us understand how agressively the policy updates
                    # PPO KL calculation
                    old_approx_kl = (-logratio).mean()
                    # Newer approach from http://joschu.net/blog/kl-approx.html
                    approx_kl = ((ratio - 1) - logratio).mean()
                    # --- Track how often the clipped objective is triggered
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                
                mb_advantages = b_advantages[mb_inds]
                # Apply PPO's advantage normalisation
                if args.norm_adv:
                    # Include small scalar 1e-8 to avoid divide by 0 error
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)


                # Setup PPO's clipped objective ---------

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                # Here we take max of negatives whereas PPO paper minimises a positive - these are equivalent!
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # PPOs clipped Value loss
                newvalue = newvalue.view(-1)
                # Unclipped loss is the squared difference between the predicted values and the actual returns
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2

                if args.clip_vloss:
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    # A more usual value loss uses the MSE between the predicted values and the actual returns
                    v_loss = 0.5 * (v_loss_unclipped).mean()

                # Entropy here measures the disorder in the action probability distribution - maximising entropy encourages exploration
                entropy_loss = entropy.mean()
                # Aim is to minimise policy loss and value loss but maximise entropy loss
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef


                # zero out the gradients before each optimization step to avoid them being accumulated across multiple batches
                optimiser.zero_grad()
                # Perform backprop
                loss.backward()
                # Apply global gradient clipping to prevent exploding gradients
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                # Update the network weights
                optimiser.step()
                
                # Save model
                # if global_epochs > 0 and global_epochs % 1000 == 0:
                #     torch.save({'epoch': epoch,
                #                 'model_state_dict': agent.state_dict(),
                #                 'optimizer_state_dict': optimiser.state_dict(),
                #                 'loss': loss}, 
                #                 f'checkpoints/{run_name}/epoch{epoch}')







            # Early stopping. Try target-kl=0.015 if using - as per 'OpenAI Spinning Up' - at batch level here but can apply at minibatch instead if preferable!
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break


        # --- Examining the variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        # Explained variance can show if value function is a good indicator of returns
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


        # Record metrics for Tensorboard
        writer.add_scalar('charts/learning_rate', optimiser.param_groups[0]['lr'], global_step)
        writer.add_scalar('losses/value_loss', v_loss.item(), global_step)
        writer.add_scalar('losses/policy_loss', pg_loss.item(), global_step)
        writer.add_scalar('losses/entropy', entropy_loss.item(), global_step)
        writer.add_scalar('losses/old_approx_kl', old_approx_kl.item(), global_step)
        writer.add_scalar('losses/approx_kl', approx_kl.item(), global_step)
        writer.add_scalar('losses/clipfrac', np.mean(clipfracs), global_step)
        writer.add_scalar('losses/explained_variance', explained_var, global_step)
        print('Steps per second:', int(global_step / (time.time() - start_time)))
        writer.add_scalar('charts/SPS', int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
