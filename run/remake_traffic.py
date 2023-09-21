import os
import sys
import torch
import datetime
import numpy as np
from common.utils_remake import plot_rewards, plot_backlogs, plot_delays, plot_average_y, plot_average_queue_lengths

#  --------------------------------基础准备--------------------------------  #
current_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(current_path)  # current_path的父路径
sys.path.append(parent_path)  # 将父路径添加到系统路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备
current_time = datetime.datetime.now().strftime("""%Y%m%d-%H%M%S""")  # 获取当前时间


class PlotConfig:
    """ 画图超参数 """

    def __init__(self) -> None:
        self.info = "Vehicle number"
        self.change = [6, 7, 8, 9]
        self.plot_path = parent_path + "/results/" + "/traffic/" + "/remake/"
        self.save = True  # 是否保存图片


read_path = parent_path + "/results/" + "/traffic/" + "weight=50,speed=7/20230821-020011/results"

reward_1 = np.load(read_path + "/rewards/train_ma_rewards_0.npy")
reward_2 = np.load(read_path + "/rewards/train_ma_rewards_1.npy")
reward_3 = np.load(read_path + "/rewards/train_ma_rewards_2.npy")
reward_4 = np.load(read_path + "/rewards/train_ma_rewards_3.npy")

backlog_1 = np.load(read_path + "/backlogs/train_ma_backlogs_0.npy")
backlog_2 = np.load(read_path + "/backlogs/train_ma_backlogs_1.npy")
backlog_3 = np.load(read_path + "/backlogs/train_ma_backlogs_2.npy")
backlog_4 = np.load(read_path + "/backlogs/train_ma_backlogs_3.npy")

delay_1 = np.load(read_path + "/delays/train_ma_delays_0.npy")
delay_2 = np.load(read_path + "/delays/train_ma_delays_1.npy")
delay_3 = np.load(read_path + "/delays/train_ma_delays_2.npy")
delay_4 = np.load(read_path + "/delays/train_ma_delays_3.npy")

y_1 = np.load(read_path + "/ys/train_ma_queue_queues_0.npy")
y_2 = np.load(read_path + "/ys/train_ma_queue_queues_1.npy")
y_3 = np.load(read_path + "/ys/train_ma_queue_queues_2.npy")
y_4 = np.load(read_path + "/ys/train_ma_queue_queues_3.npy")

queue_1 = np.load(read_path + "/queues/train_ma_queue_queues_0.npy")
queue_2 = np.load(read_path + "/queues/train_ma_queue_queues_1.npy")
queue_3 = np.load(read_path + "/queues/train_ma_queue_queues_2.npy")
queue_4 = np.load(read_path + "/queues/train_ma_queue_queues_3.npy")

vehicle_queue_1 = np.load(read_path + "/queues/train_ma_vehicle_queues_0.npy")
vehicle_queue_2 = np.load(read_path + "/queues/train_ma_vehicle_queues_1.npy")
vehicle_queue_3 = np.load(read_path + "/queues/train_ma_vehicle_queues_2.npy")
vehicle_queue_4 = np.load(read_path + "/queues/train_ma_vehicle_queues_3.npy")
cfg = PlotConfig()
# 绘制图表
plot_rewards(reward_1, reward_2, reward_3, reward_4, cfg=cfg, tag="train")
plot_backlogs(backlog_1, backlog_2, backlog_3, backlog_4, cfg=cfg, tag="train")
plot_delays(delay_1, delay_2, delay_3, delay_4, cfg=cfg, tag="train")
plot_average_y(y_1, cfg=cfg, tag_1="train", tag_2="vehicle")
plot_average_queue_lengths(vehicle_queue_1, vehicle_queue_2, vehicle_queue_3, vehicle_queue_4, cfg=cfg,
                           tag_1="train", tag_2="queue")
plot_average_queue_lengths(queue_1, queue_2, queue_3, queue_4, cfg=cfg, tag_1="train", tag_2="vehicle")
