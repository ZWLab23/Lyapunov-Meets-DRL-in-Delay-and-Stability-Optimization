import os
import sys
import torch
import datetime
from common.utils import make_dir, save_rewards, save_backlogs, save_delays
from common.utils import save_completion_ratios, save_queues, save_ys
from common.utils import plot_rewards, plot_backlogs, plot_delays
from common.utils import plot_completion_ratio, plot_average_queue_lengths, plot_average_y
from env.config import VehicularEnvConfig
from methods.SAC.task import SACConfig, TrainAndTestSAC

#  --------------------------------基础准备--------------------------------  #
current_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(current_path)  # current_path的父路径
sys.path.append(parent_path)  # 将父路径添加到系统路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备
current_time = datetime.datetime.now().strftime("""%Y%m%d-%H%M%S""")  # 获取当前时间


#  --------------------------------画图准备--------------------------------  #
class PlotConfig:
    """ 画图超参数 """

    def __init__(self, number, speed) -> None:
        self.device = device
        self.config = VehicularEnvConfig()
        self.info = "w"
        self.traffic = self.config.vehicle_number_dict.get(number)  # 读取车流信息
        self.change = list(self.config.w.values())  # 读取权重
        self.speed = self.config.rsu_compute_speed.get(speed)  # 读取RSU计算速度

        # 创建路径
        self.main_path = parent_path + "/results/" + "/weight/" + "/weight={},speed={}/" \
            .format(self.traffic, self.speed) + current_time

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


# ---------------------不同权重下的reward对比--------------------- #
def traffic_comparison():
    """ 不同权重下的reward对比 """
    cfg_1 = SACConfig(number_tag="number_1", weight_tag="weight_1", speed_tag="speed_2", stability_tag="a",
                      flow_tag="nf")
    cfg_2 = SACConfig(number_tag="number_1", weight_tag="weight_2", speed_tag="speed_2", stability_tag="a",
                      flow_tag="nf")
    cfg_3 = SACConfig(number_tag="number_1", weight_tag="weight_3", speed_tag="speed_2", stability_tag="a",
                      flow_tag="nf")
    cfg_4 = SACConfig(number_tag="number_1", weight_tag="weight_4", speed_tag="speed_2", stability_tag="a",
                      flow_tag="nf")
    plot_cfg = PlotConfig(number="number_1", speed="speed_2")  # 画图参数
    sac_mind_1 = TrainAndTestSAC(cfg_1)
    sac_mind_2 = TrainAndTestSAC(cfg_2)
    sac_mind_3 = TrainAndTestSAC(cfg_3)
    sac_mind_4 = TrainAndTestSAC(cfg_4)

    # -----------------------------------训练过程----------------------------------- #
    rewards_1, completion_ratios_1, backlogs_1, delays_1, vehicle_lengths_1, rsu_lengths_1, queue_lengths_1, \
        vehicle_ys_1, rsu_ys_1, ys_1 = sac_mind_1.train()
    rewards_2, completion_ratios_2, backlogs_2, delays_2, vehicle_lengths_2, rsu_lengths_2, queue_lengths_2, \
        vehicle_ys_2, rsu_ys_2, ys_2 = sac_mind_2.train()
    rewards_3, completion_ratios_3, backlogs_3, delays_3, vehicle_lengths_3, rsu_lengths_3, queue_lengths_3, \
        vehicle_ys_3, rsu_ys_3, ys_3 = sac_mind_3.train()
    rewards_4, completion_ratios_4, backlogs_4, delays_4, vehicle_lengths_4, rsu_lengths_4, queue_lengths_4, \
        vehicle_ys_4, rsu_ys_4, ys_4 = sac_mind_4.train()

    # 创建保存结果和模型路径的文件夹
    make_dir(plot_cfg.model_path + '/weight_1/', plot_cfg.model_path + '/weight_2/',
             plot_cfg.model_path + '/weight_3/', plot_cfg.model_path + '/weight_4/')
    make_dir(plot_cfg.reward_path, plot_cfg.backlog_path, plot_cfg.delay_path, plot_cfg.plot_path,
             plot_cfg.completion_ratio_path, plot_cfg.queue_path, plot_cfg.y_path)

    # 保存模型
    sac_mind_1.agent.save(plot_cfg.model_path + '/weight_1/')
    sac_mind_2.agent.save(plot_cfg.model_path + '/weight_2/')
    sac_mind_3.agent.save(plot_cfg.model_path + '/weight_3/')
    sac_mind_4.agent.save(plot_cfg.model_path + '/weight_4/')

    # 保存结果
    save_rewards(rewards_1, rewards_2, rewards_3, rewards_4, tag="train", path=plot_cfg.reward_path)
    save_backlogs(backlogs_1, backlogs_2, backlogs_3, backlogs_4, tag="train", path=plot_cfg.backlog_path)
    save_delays(delays_1, delays_2, delays_3, delays_4, tag="train", path=plot_cfg.delay_path)
    save_completion_ratios(completion_ratios_1, completion_ratios_2, completion_ratios_3, completion_ratios_4,
                           tag="train", path=plot_cfg.completion_ratio_path)
    save_queues(queue_lengths_1, queue_lengths_2, queue_lengths_3, queue_lengths_4, tag_1="train", tag_2="queue",
                path=plot_cfg.queue_path)
    save_queues(vehicle_lengths_1, vehicle_lengths_2, vehicle_lengths_3, vehicle_lengths_4, tag_1="train",
                tag_2="vehicle", path=plot_cfg.queue_path)
    save_queues(rsu_lengths_1, rsu_lengths_2, rsu_lengths_3, rsu_lengths_4, tag_1="train", tag_2="rsu",
                path=plot_cfg.queue_path)
    save_ys(ys_1, ys_2, ys_3, ys_4, tag_1="train", tag_2="queue", path=plot_cfg.y_path)
    save_ys(vehicle_ys_1, vehicle_ys_2, vehicle_ys_3, vehicle_ys_4, tag_1="train", tag_2="vehicle",
            path=plot_cfg.y_path)
    save_ys(rsu_ys_1, rsu_ys_2, rsu_ys_3, rsu_ys_4, tag_1="train", tag_2="rsu", path=plot_cfg.y_path)

    # 画图
    plot_rewards(rewards_1, rewards_2, rewards_3, rewards_4, cfg=plot_cfg, tag="train")
    plot_backlogs(backlogs_1, backlogs_2, backlogs_3, backlogs_4, cfg=plot_cfg, tag="train")
    plot_delays(delays_1, delays_2, delays_3, delays_4, cfg=plot_cfg, tag="train")
    plot_completion_ratio(completion_ratios_1, completion_ratios_2, completion_ratios_3, completion_ratios_4,
                          cfg=plot_cfg, tag="train")
    plot_average_queue_lengths(vehicle_lengths_1, vehicle_lengths_2, vehicle_lengths_3, vehicle_lengths_4,
                               cfg=plot_cfg, tag_1="train", tag_2="vehicle")
    plot_average_queue_lengths(rsu_lengths_1, rsu_lengths_2, rsu_lengths_3, rsu_lengths_4, cfg=plot_cfg,
                               tag_1="train", tag_2="rsu")
    plot_average_queue_lengths(queue_lengths_1, queue_lengths_2, queue_lengths_3, queue_lengths_4, cfg=plot_cfg,
                               tag_1="train", tag_2="queue")
    plot_average_y(vehicle_ys_1, vehicle_ys_2, vehicle_ys_3, vehicle_ys_4, cfg=plot_cfg, tag_1="train", tag_2="vehicle")
    plot_average_y(rsu_ys_1, rsu_ys_2, rsu_ys_3, rsu_ys_4, cfg=plot_cfg, tag_1="train", tag_2="rsu")
    plot_average_y(ys_1, ys_2, ys_3, ys_4, cfg=plot_cfg, tag_1="train", tag_2="y")

    # -----------------------------------测试过程----------------------------------- #
    # 导入模型
    sac_mind_1.agent.load(plot_cfg.model_path + '/weight_1/')
    sac_mind_2.agent.load(plot_cfg.model_path + '/weight_2/')
    sac_mind_3.agent.load(plot_cfg.model_path + '/weight_3/')
    sac_mind_4.agent.load(plot_cfg.model_path + '/weight_4/')

    rewards_1, growths_1, t_taus_1 = sac_mind_1.test()
    rewards_2, growths_2, t_taus_2 = sac_mind_2.test()
    rewards_3, growths_3, t_taus_3 = sac_mind_3.test()
    rewards_4, growths_4, t_taus_4 = sac_mind_4.test()

    # 画图
    plot_rewards(rewards_1, rewards_2, rewards_3, rewards_4, cfg=plot_cfg, tag="test")
    plot_backlogs(backlogs_1, backlogs_2, backlogs_3, backlogs_4, cfg=plot_cfg, tag="test")
    plot_delays(delays_1, delays_2, delays_3, delays_4, cfg=plot_cfg, tag="test")


if __name__ == "__main__":
    traffic_comparison()
