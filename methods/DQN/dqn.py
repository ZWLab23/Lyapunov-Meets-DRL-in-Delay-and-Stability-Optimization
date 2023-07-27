import math
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#  --------------------------------构建网络--------------------------------  #
class Net(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=128):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(n_states, hidden_dim)  # 输入层
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.linear3 = nn.Linear(hidden_dim, n_actions)  # 输出层

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        q_value = self.linear3(x)
        return q_value


#  --------------------------------经验回放--------------------------------  #
class ReplayBuffer:
    """ 经验回放池的构建 """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []  # 缓冲区
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """ 缓冲区是一个队列，容量超出时去掉开始存入的转移（transition） """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """ 采样 """
        batch = random.sample(self.buffer, batch_size)  # 随机采出mini-batch
        state, action, reward, next_state, done = zip(*batch)  # 解压
        return state, action, reward, next_state, done

    def __len__(self):
        """ 返回当前存储的数据量 """
        return len(self.buffer)


#  --------------------------------算法逻辑--------------------------------  #
class DQN:
    """ 主要更新过程 """

    def __init__(self, n_states, n_actions, cfg):
        self.batch_size = cfg.batch_size
        self.memory = ReplayBuffer(cfg.capacity)
        self.device = cfg.device

        self.n_actions = n_actions
        self.gamma = cfg.gamma  # 折扣因子

        # epsilon衰减
        self.frame_idx = 0  # epsilon的衰减计数
        self.epsilon = lambda frame_idx: cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * \
                                         math.exp(-1. * frame_idx / cfg.epsilon_decay)

        # 定义估计网络和目标网络
        self.policy_net = Net(n_states, n_actions).to(self.device)
        self.target_net = Net(n_states, n_actions).to(self.device)

        # 目标网络参数初始化
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        # 优化器设置
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)

    def choose_action(self, state):
        """ 选择动作 """
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                q_values = self.policy_net(state)
                index = q_values.max(1)[1].item()  # 选择Q值最大的动作
                action = [0 for _ in range(self.n_actions)]
                action[index] = 1
        else:
            action = [0 for _ in range(self.n_actions)]
            index = random.randrange(self.n_actions)
            action[index] = 1
        return action

    def update(self):
        """ 网络参数更新主要思路 """
        # 先从经验池D中采样
        if len(self.memory) < self.batch_size:  # 当memory中不满足一个批量时，不更新策略
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)

        # q估计值的计算
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)
        indices = torch.nonzero(action_batch)
        q_values = q_values[indices[:, 0], indices[:, 1]]
        # q目标值的计算
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss_fn = nn.MSELoss()
        loss = loss_fn(q_values, expected_q_values)

        # 梯度下降
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save(self, path):
        torch.save(self.target_net.state_dict(), path + 'dqn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path + 'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)
