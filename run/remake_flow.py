import os
import sys
import torch
import datetime
import numpy as np
from common.utils_remake import plot_rewards, plot_backlogs, plot_delays, plot_average_queue_lengths

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
        self.change = ['with node-mounted tasks', 'without node-mounted tasks']
        self.plot_path = parent_path + "/results/" + "/flow/" + "/remake/"
        self.save = True  # 是否保存图片


read_path = parent_path + "/results/" + "/flow/" + "/weight=50,speed=8,number=6/20230816-210224/results"

reward_a = np.load(read_path + "/rewards/train_ma_rewards_0.npy")
reward_noa = np.load(read_path + "/rewards/train_ma_rewards_1.npy")

backlog_a = np.load(read_path + "/backlogs/train_ma_backlogs_0.npy")
backlog_noa = np.load(read_path + "/backlogs/train_ma_backlogs_1.npy")

delay_a = np.load(read_path + "/delays/train_ma_delays_0.npy")
delay_noa = np.load(read_path + "/delays/train_ma_delays_1.npy")

queue_a = np.load(read_path + "/queues/train_ma_queue_queues_0.npy")
queue_noa = np.load(read_path + "/queues/train_ma_queue_queues_1.npy")

cfg = PlotConfig()
# 绘制图表
plot_rewards(reward_a, reward_noa, cfg=cfg, tag="train")
plot_delays(delay_a, delay_noa, cfg=cfg, tag="train")
plot_backlogs(backlog_a, backlog_noa, cfg=cfg, tag="train")
plot_average_queue_lengths(queue_a, queue_noa, cfg=cfg, tag_1="train", tag_2="queue")
