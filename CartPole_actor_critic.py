import gymnasium as gym
from gymnasium.utils.save_video import save_video
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

class VNetwork(nn.Module):
    def __init__(self):
        super(VNetwork, self).__init__()
        self.fc0 = nn.Linear(4, 32)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class PiNetwork(nn.Module):
    def __init__(self):
        super(PiNetwork, self).__init__()
        self.fc0 = nn.Linear(4, 32)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.relu(self.fc0(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
v_estermator = VNetwork().to(device)
q_optimizer = optim.SGD(v_estermator.parameters(),lr=1e-3)
pi = PiNetwork().to(device)
pi_optimizer = optim.SGD(pi.parameters(),lr=1e-4)
criter = nn.MSELoss()

def save_update(save,new,n):
    for i in range(n):
        save[i] = save[i+1]
    save[n] = new
        
def f(episode_id: int) -> bool:
    if episode_id % 25 == 0:
        return True
    elif episode_id % 25 == 1:
        return True
    elif episode_id % 25 == 2:
        return True
    elif episode_id % 25 == 3:
        return True
    elif episode_id % 25 == 4:
        return True
    else:
        return False

def run(episodes, is_training=True, render=False):
    env = gym.make('CartPole-v1', render_mode="rgb_array_list" if render else None)

    discount = 0.99
    epsilon = 1
    epsilon_decay_rate = 0.99
    rng = np.random.default_rng()

    state_save = [None]*(2001)
    action_save = [None]*(2001)
    reward_save = [None]*(2001)
    
    reward_sum = 0
    for i in range(episodes):
        print(i)
        state = env.reset()[0]
        state[2] = state[2]*10
        
        probability = pi(torch.tensor(state, dtype=torch.float32).to(device).reshape(1,-1))
        
        if rng.random() < probability[0,0]:
            action = 0
        else:
            action = 1
            
        if is_training and rng.random() < epsilon:
            action = np.random.randint(2)
            
        terminated = False
        truncated = False
        
        save_update(state_save,state.copy(),2000)
        save_update(reward_save,None,2000)
        save_update(action_save,action,2000)
        
        while(not terminated and not truncated):

            state,reward,terminated,truncated,_ = env.step(np.array(action))
            
            if terminated or truncated:
                reward = -1
            
            reward_sum += reward
            
            state[2] = state[2]*10
            
            probability = pi(torch.tensor(state, dtype=torch.float32).to(device).reshape(1,-1))
            if rng.random() < probability[0,0]:
                action = 0
            else:
                action = 1
                
            if is_training and rng.random() < epsilon:
                action = np.random.randint(2)
                
            save_update(state_save,state.copy(),2000)
            save_update(reward_save,reward,2000)
            if terminated or truncated:
                save_update(action_save,None,2000)
            else:
                save_update(action_save,action,2000)
                    
            v_mini = torch.tensor([0]*64, dtype=torch.float).to(device)
            g_mini = torch.tensor([0]*64, dtype=torch.float).to(device)
            l = 0
            for j in range(64):
                pi_optimizer.zero_grad()
                random_index = np.random.randint(2001-5)
                if not action_save[random_index] == None:
                    if not action_save[random_index+1] == None:
                        if not action_save[random_index+2] == None:
                            if not action_save[random_index+3] == None:
                                if not action_save[random_index+4] == None:
                                    G = v_estermator(torch.tensor(state_save[random_index+4], dtype=torch.float32).to(device).reshape(1,-1))[0]
                                    G = reward_save[random_index+4] + discount*G
                                else:
                                    G = v_estermator(torch.tensor(state_save[random_index+3], dtype=torch.float32).to(device).reshape(1,-1))[0]
                                G = reward_save[random_index+3] + discount*G
                            else:
                                G = v_estermator(torch.tensor(state_save[random_index+2], dtype=torch.float32).to(device).reshape(1,-1))[0]
                            G = reward_save[random_index+2] + discount*G
                        else:
                            G = v_estermator(torch.tensor(state_save[random_index+1], dtype=torch.float32).to(device).reshape(1,-1))[0]
                    else:
                        G = torch.tensor([0], dtype=torch.float).to(device)
                    G = reward_save[random_index+1] + discount*G
                    v_mini[l]=v_estermator(torch.tensor(state_save[random_index], dtype=torch.float32).to(device).reshape(1,-1))[0]
                    g_mini[l]=G
                    
                    l += 1
            if not l == 0:
                loss = criter(v_mini[:l].reshape(1,l),g_mini[:l].reshape(1,l).detach())
                
                q_optimizer.zero_grad()
                loss.backward()
                q_optimizer.step()
                
            for j in range(8):
                random_index = np.random.randint(2000)
                if not action_save[random_index] == None:
                    if not action_save[random_index+1] == None:
                        G = v_estermator(torch.tensor(state_save[random_index+1], dtype=torch.float32).to(device).reshape(1,-1))[0]
                    else:
                        G = torch.tensor([0], dtype=torch.float).to(device)
                    G = reward_save[random_index+1] + discount*G
                    V = v_estermator(torch.tensor(state_save[random_index], dtype=torch.float32).to(device).reshape(1,-1))[0]
                    probability0 = pi(torch.tensor(state_save[random_index], dtype=torch.float32).to(device).reshape(1,-1))[0,action_save[random_index]]
                    if not(probability0 > -0.00000001 and probability0 < 0.00000001):
                        probability1 = probability0.detach()
                        pi_optimizer.zero_grad()
                        (-1*(G-V).detach()*probability0/probability1).backward()
                        pi_optimizer.step()
                             
        epsilon = max(epsilon*epsilon_decay_rate, 0.0)
        
        if i%10==9:
            print('---------------------------')
            print('episode:',i)
            print('epsilon:',epsilon)
            print('reward:',reward_sum/10)
            reward_sum = 0
            
        if terminated or truncated:
            save_video(
                env.render(),
                "videos",
                episode_trigger=f,
                fps=env.metadata["render_fps"],
                step_starting_index=0,
                episode_index=i
            )
    env.close()

if __name__ == '__main__':

    run(3000, is_training=True, render=True)
