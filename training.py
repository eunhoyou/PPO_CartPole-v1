from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from PPO_vector_envs import PPO

use_cuda = False
save_weights = True

# environment hyperparameters
n_updates = 1000
n_envs = 8
n_steps_per_update = 256

# agent hyperparameters
actor_lr = 0.001
critic_lr = 0.005
clip_param = 0.2
vf_coef = 1.0
ent_coef = 0.01
gamma = 0.99
gae_lambda = 0.95

# training hymerparameters
epochs = 10
batch_size = 32

def gae(rewards, values, next_value, masks, gamma=0.99, gae_lambda=0.95):
    advantages = np.zeros_like(rewards)
    last_gae_lam = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = masks[t]
            next_values = next_value
        else:
            next_non_terminal = masks[t + 1]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
    returns = advantages + values
    return advantages, returns

def main():
    envs = gym.vector.make('CartPole-v1', num_envs=n_envs)
    
    obs_shape = envs.single_observation_space.shape[0]
    action_space = envs.single_action_space
    
    if use_cuda:
        device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    agent = PPO(obs_shape, action_space, device, actor_lr, critic_lr, n_envs, clip_param, vf_coef, ent_coef)
    
    # wrappers.RecordEpisodeStatistics: episode returns, episode lengths를 기록하는 wrapper environment.
    envs_wrapper = gym.wrappers.RecordEpisodeStatistics(envs, deque_size=n_envs * n_steps_per_update)  # deque: 최근 deque_size개의 experience만 저장되고, 새로운 에피소드가 추가될 때 가장 오래된 에피소드의 통계는 자동으로 제거.

    episode_returns = []
    actor_losses = []
    critic_losses = []
    entropies = []

    for update in tqdm(range(n_updates)):
        states = np.zeros((n_steps_per_update, n_envs, obs_shape))  # (256, 8, 4)
        actions = np.zeros((n_steps_per_update, n_envs))  # (256, 8)
        rewards = np.zeros((n_steps_per_update, n_envs))
        values = np.zeros((n_steps_per_update, n_envs))
        log_probs = np.zeros((n_steps_per_update, n_envs))
        masks = np.ones((n_steps_per_update, n_envs))  # terminated(0), not terminated(1), 종료 시점에서 그 이후의 리워드를 0으로 masking.

        state, _ = envs_wrapper.reset(seed=42)
        episode_rewards = np.zeros(n_envs)
        episode_lengths = np.zeros(n_envs)
        
        for step in range(n_steps_per_update):  # n_steps_per_update = 256
            '''
                state: (n_envs, obs_shape)
                action: (n_envs, action_dim)
                log_probs: (n_envs, )
                entropy: (n_envs, )
                value: (n_envs, 1)
                reward: (n_envs, )
            '''
            states[step] = state
            with torch.no_grad():
                
                action, log_prob, entropy, value = agent.get_action_and_value(torch.FloatTensor(state).to(device))
            
            state, reward, terminated, truncated, _ = envs_wrapper.step(action.cpu().numpy())  # Gymnasium env는 기본적으로 numpy 기반.
            
            actions[step] = action.cpu().numpy()
            log_probs[step] = log_prob.cpu().numpy()
            rewards[step] = reward
            masks[step] = 1.0 - (terminated)
            values[step] = value.cpu().numpy().squeeze()

            episode_rewards += reward
            episode_lengths += 1
            done_mask = terminated
            episode_returns.extend(episode_rewards[done_mask])
            episode_rewards *= (1 - done_mask)
            episode_lengths *= (1 - done_mask)

        with torch.no_grad():
            _, _, _, next_value = agent.get_action_and_value(torch.FloatTensor(state).to(device))
            next_value = next_value.cpu().numpy().squeeze()
        advantages, returns = gae(rewards, values, next_value, masks, gamma, gae_lambda)
        
        b_states = torch.FloatTensor(states.reshape((-1, obs_shape))).to(device)  # (n_steps_per_update * n_envs, obs_shape) -> (256*8, 4)
        b_actions = torch.FloatTensor(actions.reshape(-1)).to(device)  # (n_steps_per_update * n_envs, ) -> (256*8, )
        b_log_probs = torch.FloatTensor(log_probs.reshape(-1)).to(device)
        b_advantages = torch.FloatTensor(advantages.reshape(-1)).to(device)
        b_returns = torch.FloatTensor(returns.reshape(-1)).to(device)

        dataset = TensorDataset(b_states, b_actions, b_log_probs, b_advantages, b_returns)
        for epoch in range(epochs):
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            for batch_states, batch_actions, batch_log_probs, batch_advantages, batch_returns in data_loader:
                actor_loss, critic_loss = agent.update_parameters(
                    batch_states, batch_actions, batch_log_probs, batch_advantages, batch_returns)
                
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        entropies.append(entropy.mean().item())

        if (update + 1) % 100 == 0:
            print(f'Update {update + 1}')
            print(f'Actor Loss: {actor_loss}')
            print(f'Critic Loss: {critic_loss}')
            print(f'Mean Reward: {np.mean(episode_rewards)}')
            print(f'Mean Episode Length: {np.mean(episode_lengths)}')
            print('-----------------------------------------------')

    if save_weights:
        weights_dir = os.path.join(os.getcwd(), 'weights/PPO_CartPoleV1/')
        os.makedirs(weights_dir, exist_ok=True)
        
        save_path = os.path.join(weights_dir, 'ppo_cartpole.pth')
        torch.save(agent.state_dict(), save_path)


    rolling_length = 20
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    fig.suptitle(f'Training plots')

    axs[0, 0].set_title('Episode Returns')
    returns_moving_average = np.convolve(episode_returns, np.ones(rolling_length), mode='valid') / rolling_length
    axs[0, 0].plot(returns_moving_average)
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Return')

    axs[0, 1].set_title('Actor Loss')
    actor_loss_moving_average = np.convolve(actor_losses, np.ones(rolling_length), mode='valid') / rolling_length
    axs[0, 1].plot(actor_loss_moving_average)
    axs[0, 1].set_xlabel('Update')
    axs[0, 1].set_ylabel('Loss')

    axs[1, 0].set_title('Critic Loss')
    critic_loss_moving_average = np.convolve(critic_losses, np.ones(rolling_length), mode='valid') / rolling_length
    axs[1, 0].plot(critic_loss_moving_average)
    axs[1, 0].set_xlabel('Update')
    axs[1, 0].set_ylabel('Loss')
    
    axs[1, 1].set_title('Entropy')
    entropy_moving_average = np.convolve(entropies, np.ones(rolling_length), mode='valid') / rolling_length
    axs[1, 1].plot(entropy_moving_average)
    axs[1, 1].set_xlabel('Update')
    axs[1, 1].set_ylabel('Entropy')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()