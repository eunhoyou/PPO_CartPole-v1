import os
import gymnasium as gym
import torch
import numpy as np

from PPO_vector_envs import PPO

def main():
    env = gym.make('CartPole-v1', render_mode='human')
    
    obs_shape = env.observation_space.shape[0]
    action_space = env.action_space
    
    device = torch.device("cpu")
    
    agent = PPO(obs_shape, action_space, device)

    weights_dir = os.path.join(os.getcwd(), 'weights/PPO_CartPoleV1/')
    save_path = os.path.join(weights_dir, 'ppo_cartpole_best.pth')

    if os.path.exists(save_path):
        agent.load_state_dict(torch.load(save_path, map_location=device))
    agent.eval()

    state, _ = env.reset()
    episode_reward = 0
    done = False

    while not done:
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(torch.FloatTensor(state).unsqueeze(0).to(device))
            
        state, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
        env.render()
        
        done = terminated or truncated
        episode_reward += reward
        
    env.close()
    print(f'Episode reward: {episode_reward}')
    
if __name__ == '__main__':
    main()