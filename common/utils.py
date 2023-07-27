import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


#  --------------------------------创建路径--------------------------------  #
def make_dir(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


#  --------------------------------保存结果--------------------------------  #
def save_rewards(*rewards, tag, path):
    for i, reward in zip(range(len(rewards)), rewards):
        np.save(path + '{}_ma_rewards_{}.npy'.format(tag, i), reward)
    print('Rewards saved!')


def save_backlogs(*backlogs, tag, path):
    for i, backlog in zip(range(len(backlogs)), backlogs):
        np.save(path + '{}_ma_backlogs_{}.npy'.format(tag, i), backlog)
    print('Backlogs saved!')


def save_delays(*delays, tag, path):
    for i, delay in zip(range(len(delays)), delays):
        np.save(path + '{}_ma_delays_{}.npy'.format(tag, i), delay)
    print('Delays saved!')


def save_completion_ratios(*completion_ratios, tag, path):
    for i, completion_ratio in zip(range(len(completion_ratios)), completion_ratios):
        np.save(path + '{}_ma_completion_ratios_{}.npy'.format(tag, i), completion_ratio)
    print('Completion ratios saved!')


def save_queues(*queues, tag_1, tag_2, path):
    for i, queue in zip(range(len(queues)), queues):
        np.save(path + '{}_ma_{}_queues_{}.npy'.format(tag_1, tag_2, i), queue)
    print('Q saved!')


def save_ys(*ys, tag_1, tag_2, path):
    for i, queue in zip(range(len(ys)), ys):
        np.save(path + '{}_ma_{}_queues_{}.npy'.format(tag_1, tag_2, i), ys)
    print('y saved!')


#  --------------------------------画图--------------------------------  #
def plot_rewards(*rewards, cfg, tag="train"):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlabel('episodes', fontsize=12)
    plt.ylabel('average rewards', fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.5)
    if cfg.info == "flow":
        for i, reward in zip(range(len(rewards)), rewards):
            plt.plot(reward, label='{}'.format(cfg.change[i]))
    elif cfg.info == "algo":
        plt.plot(rewards[0], label='LQ')
        plt.plot(rewards[1], label='DQN')
        plt.plot(rewards[2], label='DDPG')
        plt.plot(rewards[3], label='Our proposal')
    else:
        for i, reward in zip(range(len(rewards)), rewards):
            plt.plot(reward, label='{}={}'.format(cfg.info, cfg.change[i]))
    plt.legend(edgecolor="black", fontsize=12)
    plt.tight_layout()
    if cfg.save:
        plt.savefig(cfg.plot_path + "{}_rewards_curve.pdf".format(tag))
    plt.show()


def plot_backlogs(*backlogs, cfg, tag="train"):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlabel('episodes', fontsize=12)
    plt.ylabel('average backlog', fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    if cfg.info == "flow":
        for i, backlog in zip(range(len(backlogs)), backlogs):
            plt.plot(backlog, label='{}'.format(cfg.change[i]))
    elif cfg.info == "algo":
        plt.plot(backlogs[0], label='LQ')
        plt.plot(backlogs[1], label='DQN')
        plt.plot(backlogs[2], label='DDPG')
        plt.plot(backlogs[3], label='Our proposal')
    else:
        for i, backlog in zip(range(len(backlogs)), backlogs):
            plt.plot(backlog, label='{}={}'.format(cfg.info, cfg.change[i]))
    plt.legend(edgecolor="black", fontsize=12)
    plt.tight_layout()
    if cfg.save:
        plt.savefig(cfg.plot_path + "{}_backlogs_curve.pdf".format(tag))
    plt.show()


def plot_delays(*delays, cfg, tag="train"):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlabel('episodes', fontsize=12)
    plt.ylabel('average delay', fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    if cfg.info == "flow":
        for i, delay in zip(range(len(delays)), delays):
            plt.plot(delay, label='{}'.format(cfg.change[i]))
    elif cfg.info == "algo":
        plt.plot(delays[0], label='LQ')
        plt.plot(delays[1], label='DQN')
        plt.plot(delays[2], label='DDPG')
        plt.plot(delays[3], label='Our proposal')
    else:
        for i, delay in zip(range(len(delays)), delays):
            plt.plot(delay, label='{}={}'.format(cfg.info, cfg.change[i]))
    plt.legend(edgecolor="black", fontsize=12)
    plt.tight_layout()
    if cfg.save:
        plt.savefig(cfg.plot_path + "{}_delays_curve.pdf".format(tag))
    plt.show()


def plot_completion_ratio(*completion_ratios, cfg, tag="train"):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.xlabel('episodes', fontsize=12)
    plt.ylabel('average task completion ratio', fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    if cfg.info == "flow":
        for i, completion_ratio in zip(range(len(completion_ratios)), completion_ratios):
            plt.plot(completion_ratio, label='{}'.format(cfg.change[i]))
    elif cfg.info == "algo":
        plt.plot(completion_ratios[0], label='LQ')
        plt.plot(completion_ratios[1], label='DQN')
        plt.plot(completion_ratios[2], label='DDPG')
        plt.plot(completion_ratios[3], label='Our proposal')
    else:
        for i, completion_ratio in zip(range(len(completion_ratios)), completion_ratios):
            plt.plot(completion_ratio, label='{}={}'.format(cfg.info, cfg.change[i]))
    plt.legend(edgecolor="black", fontsize=12)
    plt.tight_layout()
    if cfg.save:
        plt.savefig(cfg.plot_path + "{}_completion_ratio_curve.pdf".format(tag))
    plt.show()


def plot_average_queue_lengths(*average_queue_lengths, cfg, tag_1="train", tag_2="queue"):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlabel('episodes', fontsize=12)
    if tag_2 == "queue":
        plt.ylabel('Q', fontsize=12)
    elif tag_2 == "vehicle":
        plt.ylabel('$Q_v^V$', fontsize=12)
    elif tag_2 == "rsu":
        plt.ylabel('$Q_r^R$', fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.5)
    if cfg.info == "flow":
        for i, average_length in zip(range(len(average_queue_lengths)), average_queue_lengths):
            plt.plot(average_length, label='{}'.format(cfg.change[i]))
    elif cfg.info == "algo":
        plt.plot(average_queue_lengths[0], label='LQ')
        plt.plot(average_queue_lengths[1], label='DQN')
        plt.plot(average_queue_lengths[2], label='DDPG')
        plt.plot(average_queue_lengths[3], label='Our proposal')
    else:
        for i, average_length in zip(range(len(average_queue_lengths)), average_queue_lengths):
            plt.plot(average_length, label='{}={}'.format(cfg.info, cfg.change[i]))
    plt.legend(edgecolor="black", fontsize=12)
    plt.tight_layout()
    if cfg.save:
        plt.savefig(cfg.plot_path + "{}_average_{}_lengths_curve.pdf".format(tag_1, tag_2))
    plt.show()


def plot_average_y(*average_ys, cfg, tag_1="train", tag_2="y"):
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xlabel('episodes', fontsize=12)
    if tag_2 == "y":
        plt.ylabel('y', fontsize=12)
    elif tag_2 == "vehicle":
        plt.ylabel('$y_v^V$', fontsize=12)
    elif tag_2 == "rsu":
        plt.ylabel('$y_r^R$', fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.5)
    if cfg.info == "flow":
        for i, average_y in zip(range(len(average_ys)), average_ys):
            plt.plot(average_y, label='{}'.format(cfg.change[i]))
    elif cfg.info == "algo":
        plt.plot(average_ys[0], label='LQ')
        plt.plot(average_ys[1], label='DQN')
        plt.plot(average_ys[2], label='DDPG')
        plt.plot(average_ys[3], label='Our proposal')
    else:
        for i, average_y in zip(range(len(average_ys)), average_ys):
            plt.plot(average_y, label='{}={}'.format(cfg.info, cfg.change[i]))
    plt.legend(edgecolor="black", fontsize=12)
    plt.tight_layout()
    if cfg.save:
        plt.savefig(cfg.plot_path + "{}_average_{}_ys_v_curve.pdf".format(tag_1, tag_2))
    plt.show()
