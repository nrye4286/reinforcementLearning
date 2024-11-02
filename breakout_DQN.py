import collections
import cv2
import os
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from gymnasium.utils.save_video import save_video

from datetime import datetime
import copy
import random
import time

class DQNBreakout(gym.Wrapper):
    def __init__(self, render_mode='rgb_array_list', repeat=4, device='cpu'):
        env = gym.make('ALE/Breakout-v5', render_mode=render_mode, frameskip=1, repeat_action_probability=0.0)
        super(DQNBreakout, self).__init__(env)
        
        self.image_shape = (84,84)
        self.repeat = repeat
        self.lives = 5
        self.frame_buffer = []
        self.device = device
        
    def step(self, action):
        total_reward = 0
        done = False
        
        for i in range(self.repeat):
            observation, reward, done, truncacted, info = self.env.step(action)
            
            total_reward += reward
            
            current_lives = info['lives']
            
            if current_lives < self.lives:
                total_reward = total_reward - 1
                self.lives = current_lives
                
            self.frame_buffer.append(observation)
            
            if done:
                break
        
        max_frame = np.max(self.frame_buffer[-2:], axis=0)
        max_frame = self.process_observation(max_frame)
        max_frame = max_frame.to(self.device)
        
        total_reward = torch.tensor(total_reward).view(1,-1).float()
        total_reward = total_reward.to(self.device)
        
        done = torch.tensor(done).view(1,-1)
        done = done.to(self.device)
        
        return max_frame, total_reward, done, info
    
    def reset(self):
        self.frame_buffer = []
        
        observation, _ = self.env.reset()
        
        self.lives = 5
        
        observation = self.process_observation(observation)

        return observation
    
    def process_observation(self, observation):
        img = Image.fromarray(observation)
        img = img.resize(self.image_shape)
        img = img.convert("L")
        img = np.array(img)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        img = img.unsqueeze(0)
        img = img / 255.0
        
        img = img.to(self.device)
        
        return img

class AtariNet(nn.Module):
    def __init__(self, nb_action=4):
        super(AtariNet, self).__init__()
        
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(1,32,kernel_size=(8,8), stride=(4,4))
        self.conv2 = nn.Conv2d(32,64,kernel_size=(4,4), stride=(2,2))
        self.conv3 = nn.Conv2d(64,64,kernel_size=(3,3), stride=(1,1))
        
        self.flatten = nn.Flatten()
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.action_value1 = nn.Linear(3136, 1024)
        self.action_value2 = nn.Linear(1024, 1024)
        self.action_value3 = nn.Linear(1024, nb_action)
        
        self.state_value1 = nn.Linear(3136, 1024)
        self.state_value2 = nn.Linear(1024, 1024)
        self.state_value3 = nn.Linear(1024, 1)
        
    def forward(self, x):
        x = torch.Tensor(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        
        state_value = self.relu(self.state_value1(x))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value2(state_value))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value3(state_value))

        action_value = self.relu(self.action_value1(x))
        action_value = self.dropout(action_value)
        action_value = self.relu(self.action_value2(action_value))
        action_value = self.dropout(action_value)
        action_value = self.relu(self.action_value3(action_value))

        output = state_value + (action_value - action_value.mean())
        
        return output
    
    def save_the_model(self, weights_filename='models/latest.pt'):
        if not os.path.exists("models"):
            os.makedirs("models")
        torch.save(self.state_dict(), weights_filename)
        
    def load_the_model(self, weights_filename='models/latest.pt'):
        try:
            self.load_state_dict(torch.load(weights_filename, map_location=device))
            print(f"Successfully loaded weights file {weights_filename}")
        except:
            print(f"No weights file available at {weights_filename}")

class ReplayMemory:
    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device
        self.memory_max_report = 0
        
    def insert(self, transition):
        transition = [item.to('cpu') for item in transition]
        
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory.remove(self.memory[0])
            self.memory.append(transition)
    
    def sample(self, batch_size=32):
        assert self.can_sample(batch_size)
        
        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        return [torch.cat(items).to(self.device) for items in batch]
    
    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10

    def __len__(self):
        return len(self.memory)

def f(episode_id: int) -> bool:
    return True

class Agent:
    def __init__(self, model, device='cpu', epsilon=1.0, min_epsilon=0.1, nb_warmup=10000,nb_action=None, memory_capacity=10000,
                 batch_size=32, learning_rate=0.00025):
        self.memory = ReplayMemory(device=device, capacity=memory_capacity)
        self.model = model
        self.target_model = copy.deepcopy(model).eval()
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = 1 - (((epsilon - min_epsilon) / nb_warmup) * 2)
        self.batch_size = batch_size
        self.model.to(device)
        self.target_model.to(device)
        self.gamma = 0.99
        self.nb_action = nb_action
        
        self.optimizer = optim.AdamW(model.parameters(),lr=learning_rate)
        
        print(f"Starting epsilon is {self.epsilon}")
        print(f"Epsilon dacay is {self.epsilon_decay}")
        
    def get_action(self, state):
        if torch.rand(1) < self.epsilon:
            return torch. randint(self.nb_action, (1,1))
        else:
            av = self.model(state).detach()
            return torch.argmax(av, dim=1, keepdim=True)
        
    def train(self, env, epochs):
        stats = {'Returns': [], 'AvgReturns': [], 'EpsilonCheckpoint': []}
        
        plotter = LivePlot()
        
        for epoch in range(1,epochs + 1):
            state = env.reset()
            done = False
            ep_return = 0
            
            while not done:
                action = self.get_action(state)
                
                next_state, reward, done, info = env.step(action)
                
                self.memory.insert([state, action, reward, done, next_state])
                
                if self.memory.can_sample(self.batch_size):
                    state_b, action_b, reward_b, done_b, next_state_b = self.memory.sample(self.batch_size)
                    qsa_b = self.model(state_b).gather(1,action_b)
                    next_qsa_b = self.target_model(next_state_b)
                    next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]
                    target_b = reward_b + ~done_b * self.gamma * next_qsa_b
                    loss = F.mse_loss(qsa_b, target_b)
                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                state = next_state
                ep_return += reward.item()
                
            stats['Returns'].append(ep_return)
            
            if self.epsilon > self.min_epsilon:
                self.epsilon = self.epsilon * self.epsilon_decay
            
            if epoch  % 10 == 0:
                self.model.save_the_model()
                print(" ")
                
                average_returns = np.mean(stats['Returns'][-100:])
                stats['AvgReturns'].append(average_returns)
                stats['EpsilonCheckpoint'].append(self.epsilon)
                
                if(len(stats['Returns'])) > 100:
                    print(f"Epoch: {epoch} - Average Return: {np.mean(stats['Returns'][-100:])} - Epsilon: {self.epsilon}")
                    print(f"memory: {self.memory.__len__()}")
                else:
                    print(f"Epoch: {epoch} - Average Return: {np.mean(stats['Returns'][-1:])} - Epsilon: {self.epsilon}")
                    print(f"memory: {self.memory.__len__()}")

            if epoch % 100 == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                plotter.update_plot(stats)

                save_video(
                env.render(),
                "videos",
                episode_trigger=f,
                fps=24,
                step_starting_index=0,
                episode_index=epoch)                
                
            if epoch % 1000 == 0:
                self.model.save_the_model(f"models/model_iter_{epoch}.pt")

        return stats
    
    def test(self, env):
        for epoch in range(1,3):
            state = env.reset()
            done = False
            for _ in range(1000):
                time.sleep(0.01)
                action = self.get_action(state)
                state, reward, done, info = env.step(action)
                if done:
                    break

class LivePlot():
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Epoch x 10")
        self.ax.set_ylabel("Returns")
        self.ax.set_title("returns over Epochs")
        
        self.data = None
        self.eps_data = None
        
        self.epochs = 0
        
    def update_plot(self, stats):
        self.data = stats['AvgReturns']
        self.eps_data = stats['EpsilonCheckpoint']
        
        self.epochs = len(self.data)
        
        self.ax.clear()
        self.ax.set_xlim(0,self.epochs)
        
        self.ax.plot(self.data, 'b-', label='Returns')
        self.ax.plot(self.eps_data, 'r-', label='Epsilon')
        self.ax.legend(loc='upper left')
        
        if not os.path.exists('plots'):
            os.makedirs('plots')
            
        current_date = datetime.now().strftime('%Y-%m-%d')
            
        self.fig.savefig(f'plots/plot_{current_date}')

os.environ['KMP_DUPLICATE_OK'] = 'TRUE'

device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

environment = DQNBreakout(device=device)
model = AtariNet(nb_action=4).to(device)

model.load_the_model()

agent = Agent(model=model,
              device=device,
              epsilon=1,
              nb_warmup=5000,
              nb_action=4,
              learning_rate=0.00001,
              memory_capacity=600000,
              batch_size=64)

agent.train(env=environment, epochs=10000000)
