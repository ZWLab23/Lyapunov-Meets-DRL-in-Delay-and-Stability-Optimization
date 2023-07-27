import copy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备


#  --------------------------------构建网络--------------------------------  #
class SoftValueNet(nn.Module):
    """ soft状态值函数网络 """

    def __init__(self, n_states, hidden_dim, init_w=3e-3):
        super(SoftValueNet, self).__init__()

        # 三层线性层
        self.linear1 = nn.Linear(n_states, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # 初始化参数
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        """ 前向传播 """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        value = self.linear3(x)
        return value


class SoftQNet(nn.Module):
    """ soft动作值函数网络 """

    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3):
        super(SoftQNet, self).__init__()

        self.linear1 = nn.Linear(n_states + n_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)  # 将状态和动作并成一个tensor作为神经网络的输入
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        q_value = self.linear3(x)
        return q_value


class PolicyNet(nn.Module):
    """ 策略函数网络 """

    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNet, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(n_states, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        # 定义一个新网络f输出均值和标准差
        self.mean_linear = nn.Linear(hidden_dim, n_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_dim, n_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        # 计算均值
        mean = F.softmax(self.mean_linear(x))

        # 计算标准差
        log_std = F.softmax(self.log_std_linear(x))
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.softmax(z, dim=1).squeeze(0)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.softmax(z, dim=1).squeeze(0)

        action = action.detach().cpu().numpy()
        return action


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


#  --------------------------------策略迭代--------------------------------  #
class SAC:
    """ SAC算法的主要更新逻辑 """

    def __init__(self, n_states, n_actions, cfg) -> None:
        self.batch_size = cfg.batch_size
        self.memory = ReplayBuffer(cfg.capacity)
        self.device = cfg.device

        # soft价值函数网络（原网络和目标网络）
        self.value_net = SoftValueNet(n_states, cfg.hidden_dim).to(self.device)
        self.target_value_net = SoftValueNet(n_states, cfg.hidden_dim).to(self.device)

        # soft动作价值函数网络
        self.soft_q_net = SoftQNet(n_states, n_actions, cfg.hidden_dim).to(self.device)

        # 策略网络
        self.policy_net = PolicyNet(n_states, n_actions, cfg.hidden_dim).to(self.device)

        # 优化器设置
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=cfg.value_lr)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=cfg.soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.policy_lr)

        # 复制参数
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        # 最小二乘法
        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()

    def update(self, gamma=0.99, mean_lambda=1e-3, std_lambda=1e-3, z_lambda=0, soft_tau=1e-2):
        """ 网络参数更新主要思路 """

        # 先从经验池D中采样
        if len(self.memory) < self.batch_size:  # 避免出现取多
            return
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        expected_q_value = self.soft_q_net(state, action)
        expected_value = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)

        # Q网络的损失计算
        target_value = self.target_value_net(next_state)
        next_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach())

        # V网络的损失计算
        expected_new_q_value = self.soft_q_net(state, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = self.value_criterion(expected_value, next_value.detach())

        # 策略网络的损失计算
        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss = std_lambda * log_std.pow(2).mean()
        z_loss = z_lambda * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        # 分别对三个网络进行梯度下降
        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # 目标网络参数软更新
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

    def save(self, path):
        torch.save(self.value_net.state_dict(), path + "sac_value")
        torch.save(self.value_optimizer.state_dict(), path + "sac_value_optimizer")

        torch.save(self.soft_q_net.state_dict(), path + "sac_soft_q")
        torch.save(self.soft_q_optimizer.state_dict(), path + "sac_soft_q_optimizer")

        torch.save(self.policy_net.state_dict(), path + "sac_policy")
        torch.save(self.policy_optimizer.state_dict(), path + "sac_policy_optimizer")

    def load(self, path):
        self.value_net.load_state_dict(torch.load(path + "sac_value"))
        self.value_optimizer.load_state_dict(torch.load(path + "sac_value_optimizer"))
        self.target_value_net = copy.deepcopy(self.value_net)

        self.soft_q_net.load_state_dict(torch.load(path + "sac_soft_q"))
        self.soft_q_optimizer.load_state_dict(torch.load(path + "sac_soft_q_optimizer"))

        self.policy_net.load_state_dict(torch.load(path + "sac_policy"))
        self.policy_optimizer.load_state_dict(torch.load(path + "sac_policy_optimizer"))
