import os
import sys
import torch
import datetime
import numpy as np
from common.utils_remake import plot_rewards, plot_completion_ratio

#  --------------------------------基础准备--------------------------------  #
current_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(current_path)  # current_path的父路径
sys.path.append(parent_path)  # 将父路径添加到系统路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备
current_time = datetime.datetime.now().strftime("""%Y%m%d-%H%M%S""")  # 获取当前时间


class PlotConfig:
    """ 画图超参数 """

    def __init__(self) -> None:
        self.info = "flow"
        self.change = ['LQ', 'DQN', 'DDPG', 'Our Proposal']
        self.plot_path = parent_path + "/results/" + "/algo/" + "/remake/"
        self.save = True  # 是否保存图片


read_path = parent_path + "/results/" + "/algo/" + "/weight=50,speed=8,number=4/20230723-230256/results"

reward_lq = np.load(read_path + "/rewards/train_ma_rewards_0.npy")
reward_dqn = np.load(read_path + "/rewards/train_ma_rewards_1.npy")
reward_ddpg = np.load(read_path + "/rewards/train_ma_rewards_2.npy")
reward_sac = np.load(read_path + "/rewards/train_ma_rewards_3.npy")

completion_ratio_lq = np.load(read_path + "/completion_ratios/train_ma_completion_ratios_0.npy")
completion_ratio_dqn = np.load(read_path + "/completion_ratios/train_ma_completion_ratios_1.npy")
completion_ratio_ddpg = np.load(read_path + "/completion_ratios/train_ma_completion_ratios_2.npy")
completion_ratio_sac = np.load(read_path + "/completion_ratios/train_ma_completion_ratios_3.npy")

cfg = PlotConfig()
# 绘制图表
plot_rewards(reward_lq, reward_dqn, reward_ddpg, reward_sac, cfg=cfg, tag="train")
plot_completion_ratio(completion_ratio_lq, completion_ratio_dqn, completion_ratio_ddpg, completion_ratio_sac, cfg=cfg, tag='train')
