import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import matplotlib.pyplot as plt
from gymnasium.utils.save_video import save_video

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

def init_weights(m):
    if isinstance(m, nn.Linear): # nn.Linear이면 True
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)
        

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential( #행동의 차원: num_output, 각각 평균을 반환, 표준편차랑 같이 해서 확률적으로 뽑아
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std) # 표준편차를 학습시킴.
        
        self.apply(init_weights) # __init__ 뒤에 함수 넣어서 실행시키면 가중치 초기화 할 수 있음
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu) # 왜 지수형태로 사용하는지 의문. 나중에 분석 ㄱㄱ, 탐험 너무 안하길래 상수 추가해줌.
        dist  = Normal(mu, std)
        return dist, value

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        
def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.15): # 20개 단위로 업데이트에 패널티 주면서 학습. 20개를 5개씩 4번 학습하는데 이때
    for _ in range(ppo_epochs):                                                                               # old정책과 많이 차이가 안나도록 함.
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action) # 밀도 반환.

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.003 * entropy #정책이 가치보다 천천히 변하도록, 탐험 장려하도록 엔트로피 추가

            optimizer.zero_grad() # 분포에서 행동을 샘플링한 뒤, 그 행동의 밀도를 계산하는것에는 랜덤성이 없고 그래프로 구현이 가능 따라서 역전파 가능능
            loss.backward()
            optimizer.step()
            
def f(episode_id: int) -> bool:
    if episode_id % 10 == 0: return True
    else: return False
envs = [gym.make('Hopper-v4', render_mode='rgb_array_list')]+[gym.make('Hopper-v4', render_mode=None)]*(16-1)

envs_state = [[envs[i].reset()[0]] for i in range(16)]
whichs = list(range(16))

num_inputs  = envs[0].observation_space.shape[0]
num_outputs = envs[0].action_space.shape[0]

#Hyper params:
hidden_size      = 64
lr               = 5e-5 #무작정 Lr낮추는 짓 안하려고 PPO같은 알고리즘 쓰는거임.
num_steps        = 20 #num_step * len(envs)개마다 학습을 시킴
mini_batch_size  = 32 #근데 그 num_step * len(envs)개를 순서대로 하는게 아니고, 32개씩 10번 랜덤하게 뽑아서 학습시킴.
ppo_epochs       = 10

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)

def train():
    done_count = 0
    reward_sum = 0
    while True:
        total_log_probs = []
        total_states    = []
        total_actions   = []
        total_returns   = []
        total_advantage = []
        for env, state, which in zip(envs,envs_state,whichs):
            log_probs = []
            values    = []
            states    = []
            actions   = []
            rewards   = []
            masks     = []
            entropy = 0
            done = False
            
            for _ in range(num_steps):
                state[0] = torch.FloatTensor(state[0]).to(device).unsqueeze(dim=0)
                dist, value = model(state[0])

                action = dist.sample()
                next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy().squeeze())
                if which == 0:
                    reward_sum += reward
                if terminated or truncated:
                    done = True
                
                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()
                
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(device))
                masks.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(device))
                
                states.append(state[0])
                actions.append(action)
                
                state[0] = next_state
                if done:
                    if which == 0:
                        print(done_count, reward_sum)
                        
                        save_video(
                        env.render(),
                        "videos",
                        episode_trigger=f,
                        fps=env.metadata["render_fps"],
                        step_starting_index=0,
                        episode_index=done_count)
                        
                        done_count += 1
                        reward_sum = 0
                    state[0],_ = env.reset()
                    break

            next_state = torch.FloatTensor(next_state).to(device)
            _, next_value = model(next_state.unsqueeze(dim=0))
            returns = compute_gae(next_value, rewards, masks, values)
            
            returns   = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values    = torch.cat(values).detach()
            states    = torch.cat(states)
            actions   = torch.cat(actions)
            advantage = returns - values

            total_returns.append(returns)
            total_log_probs.append(log_probs)
            total_states.append(states)
            total_actions.append(actions)
            total_advantage.append(advantage)
            
        total_returns = torch.cat(total_returns)
        total_log_probs = torch.cat(total_log_probs)
        total_states = torch.cat(total_states)
        total_actions = torch.cat(total_actions)
        total_advantage = torch.cat(total_advantage)
            
        ppo_update(ppo_epochs, mini_batch_size, total_states, total_actions, total_log_probs, total_returns, total_advantage)
        torch.save(model.state_dict(),r"G:\내 드라이브\Colab Notebooks\gymnasium\model.pt")


train()

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)

model.load_state_dict(torch.load(r"G:\내 드라이브\Colab Notebooks\gymnasium\model.pt"))

def test(i):
    env = gym.make('Hopper-v4', render_mode="rgb_array_list")
    
    state = env.reset()[0]
    
    for j in range(1000):
        state = torch.FloatTensor(state).to(device).unsqueeze(dim=0)
        dist, value = model(state)

        action = dist.sample()
        next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy().squeeze())

        state = next_state
        

        
    save_video(
    env.render(),
    "videos",
    episode_trigger=f,
    fps=env.metadata["render_fps"],
    step_starting_index=0,
    episode_index=i
)

for i in range(10):
    test(i)
