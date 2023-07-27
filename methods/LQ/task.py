import torch
from methods.LQ.lq import LQ
from env.environment import LyapunovModel

algo_name = "LQ"  # 算法名称
env_name = "LyapunovModel"  # 环境名称
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备


class LQConfig:
    """ 算法超参数 """

    def __init__(self, number_tag, weight_tag, speed_tag, stability_tag, flow_tag) -> None:
        self.device = device
        self.algo_name = algo_name
        self.env_name = env_name
        # 训练设置
        self.train_eps = 1000
        self.test_eps = 20
        self.max_steps = 1000  # 每回合的最大步数

        self.number_tag = number_tag
        self.weight_tag = weight_tag
        self.speed_tag = speed_tag
        self.stability_tag = stability_tag
        self.flow_tag = flow_tag


class TrainAndTestLQ(object):
    """ 最短队列算法 """

    def __init__(self, cfg):
        """ 创建环境和智能体 """
        self.cfg = cfg
        self.env = LyapunovModel(number_tag=self.cfg.number_tag, weight_tag=self.cfg.weight_tag,
                                 speed_tag=self.cfg.speed_tag, stability_tag=self.cfg.stability_tag,
                                 flow_tag=self.cfg.flow_tag)
        self.env.reset()
        n_states = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.shape[0]
        self.agent = LQ(n_states, n_actions, cfg)

    def train(self):
        print("Start training!")
        print(f"Env:{self.cfg.env_name}, Algo:{self.cfg.algo_name}, Device:{self.cfg.device}")
        rewards = []  # 记录所有回合的滑动平均奖励
        completion_ratios = []
        backlogs = []
        delays = []
        vehicle_queue_lengths = []
        rsu_queue_lengths = []
        queue_lengths = []
        vehicle_ys = []
        rsu_ys = []
        ys = []
        for i_ep in range(self.cfg.train_eps):
            ep_completed = 0
            ep_backlog = 0
            ep_reward = 0  # 记录一回合内的奖励
            ep_delay = 0
            ep_vehicle_queue_length = 0
            ep_rsu_queue_length = 0
            ep_queue_length = 0
            ep_vehicle_y = 0
            ep_rsu_y = 0
            ep_y = 0
            state = self.env.reset()  # 重置环境，返回初始状态
            for i_step in range(self.cfg.max_steps):
                action = self.agent.choose_action(state)
                next_state, reward, backlog, delay, done, queue_v, y_v, queue_r, y_r, queue, y = self.env.step(action)
                state = next_state
                ep_reward += reward
                ep_backlog += backlog
                ep_delay += delay
                ep_vehicle_queue_length += queue_v
                ep_rsu_queue_length += queue_r
                ep_queue_length += queue
                ep_vehicle_y += y_v
                ep_rsu_y += y_r
                ep_y += y

                if reward > self.env.config.reward_threshold.get(self.env.weight_tag):
                    ep_completed += 1
                if done:
                    break

            rewards.append(0.9 * rewards[-1] + 0.1 * ep_reward) if rewards else rewards.append(ep_reward)
            backlogs.append(0.9 * backlogs[-1] + 0.1 * ep_backlog) if backlogs else backlogs.append(ep_backlog)
            delays.append(0.9 * delays[-1] + 0.1 * ep_delay) if delays else delays.append(ep_delay)

            completion_ratio = ep_completed / (self.env.config.end_time + 1)
            completion_ratios.append(0.6 * completion_ratios[-1] + 0.4 * completion_ratio) \
                if completion_ratios else completion_ratios.append(completion_ratio)

            average_ep_vehicle_queue_length = ep_vehicle_queue_length / (self.env.config.end_time + 1)
            average_ep_rsu_queue_length = ep_rsu_queue_length / (self.env.config.end_time + 1)
            average_ep_queue_length = ep_queue_length / (self.env.config.end_time + 1)
            average_ep_vehicle_y = ep_vehicle_y / (self.env.config.end_time + 1)
            average_ep_rsu_y = ep_rsu_y / (self.env.config.end_time + 1)
            average_ep_y = ep_y / (self.env.config.end_time + 1)

            vehicle_queue_lengths.append(0.9 * vehicle_queue_lengths[-1] + 0.1 * average_ep_vehicle_queue_length) \
                if vehicle_queue_lengths else vehicle_queue_lengths.append(average_ep_vehicle_queue_length)
            rsu_queue_lengths.append(0.9 * rsu_queue_lengths[-1] + 0.1 * average_ep_rsu_queue_length) \
                if rsu_queue_lengths else rsu_queue_lengths.append(average_ep_rsu_queue_length)
            queue_lengths.append(0.9 * queue_lengths[-1] + 0.1 * average_ep_queue_length) \
                if queue_lengths else queue_lengths.append(average_ep_queue_length)
            vehicle_ys.append(0.9 * vehicle_ys[-1] + 0.1 * average_ep_vehicle_y) \
                if vehicle_ys else vehicle_ys.append(ep_vehicle_y)
            rsu_ys.append(0.9 * rsu_ys[-1] + 0.1 * average_ep_rsu_y) if rsu_ys else rsu_ys.append(ep_rsu_y)
            ys.append(0.9 * ys[-1] + 0.1 * average_ep_y) if ys else ys.append(ep_y)

            if (i_ep + 1) % 1 == 0:
                print(f"Episode:{i_ep + 1}/{self.cfg.train_eps}, Reward:{ep_reward:.3f}, "
                      f"Completed:{completion_ratio: .3f}")
        print("Finish training!")
        return rewards, completion_ratios, backlogs, delays, vehicle_queue_lengths, rsu_queue_lengths, queue_lengths, \
            vehicle_ys, rsu_ys, ys

    def test(self):
        print("Test Start!")
        print(f"Env:{self.cfg.env_name}, Algo:{self.cfg.algo_name}, Device:{self.cfg.device}")
        rewards = []  # 记录所有回合的奖励
        backlogs = []
        delays = []
        for i_ep in range(self.cfg.test_eps):
            ep_reward = 0
            ep_backlog = 0
            ep_delay = 0
            state = self.env.reset()
            for i_step in range(self.cfg.max_steps):
                action = self.agent.choose_action(state)
                next_state, reward, backlog, delay, done, queue_v, y_v, queue_r, y_r, queue, y = self.env.step(action)
                state = next_state
                ep_reward += reward
                ep_backlog += backlog
                ep_delay += delay
                if done:
                    break
            rewards.append(0.9 * rewards[-1] + 0.1 * ep_reward) if rewards else rewards.append(ep_reward)
            backlogs.append(0.9 * backlogs[-1] + 0.1 * ep_backlog) if backlogs else backlogs.append(ep_backlog)
            delays.append(0.9 * delays[-1] * 0.1 * ep_delay) if delays else delays.append(ep_delay)

            print(f"Episode:{i_ep + 1}/{self.cfg.test_eps}, Reward:{ep_reward:.1f}")
        print("Test Finish!")
        return rewards, backlogs, delays
