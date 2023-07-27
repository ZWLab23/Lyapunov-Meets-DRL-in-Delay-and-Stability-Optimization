import os
import sys
import torch
import datetime
from common.utils import make_dir, save_rewards, save_backlogs, save_delays
from common.utils import save_completion_ratios, save_queues, save_ys
from common.utils import plot_rewards, plot_backlogs, plot_delays
from common.utils import plot_completion_ratio, plot_average_queue_lengths, plot_average_y
from env.config import VehicularEnvConfig
from methods.LQ.task import LQConfig, TrainAndTestLQ
from methods.DQN.task import DQNConfig, TrainAndTestDQN
from methods.SAC.task import SACConfig, TrainAndTestSAC
from methods.DDPG.task import DDPGConfig, TrainAndTestDDPG

#  --------------------------------基础准备--------------------------------  #
current_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(current_path)  # current_path的父路径
sys.path.append(parent_path)  # 将父路径添加到系统路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备
current_time = datetime.datetime.now().strftime("""%Y%m%d-%H%M%S""")  # 获取当前时间


#  --------------------------------画图准备--------------------------------  #
class PlotConfig:
    """ 画图超参数 """

    def __init__(self, weight, speed, number) -> None:
        self.device = device
        self.config = VehicularEnvConfig()
        self.info = "algo"
        self.weight = self.config.w.get(weight)  # 读取权重
        self.traffic = self.config.vehicle_number_dict.get(number)  # 读取车流信息
        self.speed = self.config.rsu_compute_speed.get(speed)  # 读取RSU计算速度
        # 创建路径
        self.main_path = parent_path + "/results/" + "/algo/" + "/weight={},speed={},number={}/" \
            .format(self.weight, self.speed, self.traffic) + current_time

        # 模型保存路径
        self.model_path = self.main_path + '/models/'

        # 结果保存路径
        self.result_path = self.main_path + '/results/'
        self.plot_path = self.result_path + '/plots/'
        self.reward_path = self.result_path + '/rewards/'
        self.backlog_path = self.result_path + '/backlogs/'
        self.delay_path = self.result_path + '/delays/'
        self.completion_ratio_path = self.result_path + '/completion_ratios/'
        self.queue_path = self.result_path + '/queues/'
        self.y_path = self.result_path + '/ys/'

        self.save = True  # 是否保存图片


# --------------------------------对比结果-------------------------------- #
def algo_comparison():
    """ 不同车辆数量下的reward对比 """
    plot_cfg = PlotConfig(weight="weight_1", speed="speed_2", number="number_1")  # 画图参数
    cfg_ddpg = DDPGConfig(number_tag="number_1", weight_tag="weight_1", speed_tag="speed_2", stability_tag="a")

    ddpg_mind = TrainAndTestDDPG(cfg_ddpg)

    # -----------------------------------训练过程----------------------------------- #
    # 创建保存结果和模型路径的文件
    make_dir(plot_cfg.model_path + '/DDPG/')
    make_dir(plot_cfg.reward_path, plot_cfg.backlog_path, plot_cfg.delay_path, plot_cfg.plot_path,
             plot_cfg.completion_ratio_path, plot_cfg.queue_path, plot_cfg.y_path)

    rewards_ddpg, completion_ratios_ddpg, backlogs_ddpg, delays_ddpg, vehicle_lengths_ddpg, rsu_lengths_ddpg, \
        queue_lengths_ddpg, vehicle_ys_ddpg, rsu_ys_ddpg, ys_ddpg = ddpg_mind.train()

    # 保存模型
    ddpg_mind.agent.save(plot_cfg.model_path + '/DDPG/')

    # 保存结果
    save_rewards(rewards_ddpg, tag="train", path=plot_cfg.reward_path)
    save_backlogs(backlogs_ddpg, tag="train", path=plot_cfg.backlog_path)
    save_delays(delays_ddpg, tag="train", path=plot_cfg.delay_path)

    # 画图
    plot_rewards(rewards_ddpg, cfg=plot_cfg, tag="train")
    plot_backlogs(backlogs_ddpg, cfg=plot_cfg, tag="train")
    plot_delays(delays_ddpg, cfg=plot_cfg, tag="train")

    # -----------------------------------测试过程----------------------------------- #
    # 导入模型
    ddpg_mind.agent.load(plot_cfg.model_path + '/DDPG/')

    rewards_lq, backlogs_lq, delays_lq = lq_mind.test()
    rewards_dqn, backlogs_dqn, delays_dqn = dqn_mind.test()
    rewards_sac, backlogs_sac, delays_sac = sac_mind.test()
    rewards_ddpg, backlogs_ddpg, delays_ddpg = ddpg_mind.test()

    # 画图
    plot_rewards(rewards_lq, rewards_dqn, rewards_sac, rewards_ddpg, cfg=plot_cfg, tag="test")
    plot_backlogs(backlogs_lq, backlogs_dqn, backlogs_sac, backlogs_ddpg, cfg=plot_cfg, tag="test")
    plot_delays(delays_lq, delays_dqn, delays_sac, delays_ddpg, cfg=plot_cfg, tag="test")


if __name__ == "__main__":
    algo_comparison()
