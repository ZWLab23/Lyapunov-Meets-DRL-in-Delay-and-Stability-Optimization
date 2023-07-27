import gym
import numpy as np

from gym import spaces
from typing import Optional, List
from env.datastruct import VehicleList, RSUList, TimeSlot, Function
from env.config import VehicularEnvConfig


class LyapunovModel(gym.Env):
    metadata = {'render.modes': []}

    def __init__(
            self,
            number_tag: str,
            weight_tag: str,
            speed_tag: str,
            stability_tag: str,
            env_config: Optional[VehicularEnvConfig] = None,
            time_slot: Optional[TimeSlot] = None,
            vehicle_list: Optional[VehicleList] = None,
            rsu_list: Optional[RSUList] = None
    ):
        self.config = env_config or VehicularEnvConfig()  # 环境参数
        self.timeslot = time_slot or TimeSlot(start=self.config.start_time, end=self.config.end_time)
        self.w = self.config.w.get(weight_tag)
        self.weight_tag = weight_tag
        self.rsu_compute_speed = self.config.rsu_compute_speed.get(speed_tag)
        self.rsu_number = self.config.rsu_number
        self.vehicle_number = self.config.vehicle_number_dict.get(number_tag)
        self.stability = stability_tag
        # 车辆与RSU的初始化
        self.vehicle_list = vehicle_list or VehicleList(
            road_range=self.config.road_range,
            min_init_queue_length=self.config.min_vehicle_init_queue_length,
            max_init_queue_length=self.config.max_vehicle_init_queue_length,
            min_drive_speed=self.config.min_vehicle_drive_speed,
            max_drive_speed=self.config.max_vehicle_drive_speed,
            min_compute_speed=self.config.min_vehicle_compute_speed,
            max_compute_speed=self.config.max_vehicle_compute_speed,
            vehicle_number=self.vehicle_number
        )
        self.rsu_list = rsu_list or RSUList(
            min_init_queue_length=self.config.min_rsu_init_queue_length,
            max_init_queue_length=self.config.max_rsu_init_queue_length,
            compute_speed=self.rsu_compute_speed,
            rsu_number=self.rsu_number
        )
        # 定义动作和状态空间
        action_low = np.zeros(self.vehicle_number + self.rsu_number + 1)
        action_high = np.ones(self.vehicle_number + self.rsu_number + 1)
        self.action_space = spaces.Box(action_low, action_high, dtype=np.float32)
        # 定义状态空间
        observation_low = np.zeros(self.rsu_number + self.vehicle_number)
        self.observation_high = np.concatenate((np.full(self.rsu_number, 1e+6), np.full(self.vehicle_number, 1e+6)))
        self.observation_space = spaces.Box(observation_low, self.observation_high, dtype=np.float32)
        self.state = None
        self.function = None

    def _rsu_perception(self) -> np.ndarray:
        """ 通过 RSU 感知获得道路状态信息 """
        rsu_state = [rsu.task_queue for rsu in self.rsu_list.rsu_list]
        vehicle_state = [vehicle.task_queue for vehicle in self.vehicle_list.vehicle_list]
        self.state = np.concatenate([rsu_state, vehicle_state])
        return np.array(self.state, dtype=np.float32)

    def _function_generator(self) -> object:
        """ 产生我们关注的任务 """
        input_size = float(np.random.uniform(self.config.min_in_datasize, self.config.max_in_datasize))
        difficulty = float(np.random.randint(8, 10) / 10)
        output_size = float(np.random.uniform(self.config.min_out_datasize, self.config.max_out_datasize))
        delay_threshold = int(np.random.uniform(self.config.min_delay_threshold, self.config.max_delay_threshold))
        new_function = Function(input_size, difficulty, output_size, delay_threshold)
        return new_function

    def _reset_road(self) -> None:
        """ 重置RSU队列，车辆队列 """
        self.vehicle_list = VehicleList(road_range=self.config.road_range,
                                        min_init_queue_length=self.config.min_vehicle_init_queue_length,
                                        max_init_queue_length=self.config.max_vehicle_init_queue_length,
                                        min_drive_speed=self.config.min_vehicle_drive_speed,
                                        max_drive_speed=self.config.max_vehicle_drive_speed,
                                        min_compute_speed=self.config.min_vehicle_compute_speed,
                                        max_compute_speed=self.config.max_vehicle_compute_speed,
                                        vehicle_number=self.vehicle_number)
        self.rsu_list = RSUList(min_init_queue_length=self.config.min_rsu_init_queue_length,
                                max_init_queue_length=self.config.max_rsu_init_queue_length,
                                compute_speed=self.rsu_compute_speed,
                                rsu_number=self.rsu_number)

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        self.timeslot.reset()  # 重置时间
        self._reset_road()  # 重置道路
        self.function = self._function_generator()  # 新任务
        self.state = self._rsu_perception()  # 读取状态
        return np.array(self.state, dtype=np.float32)

    def _task_execute(self) -> List[float]:
        """ 上一个时隙的任务计算 """
        b_tau = [rsu.task_execute() for rsu in self.rsu_list.rsu_list]
        b_tau.extend([vehicle.task_execute() for vehicle in self.vehicle_list.vehicle_list])
        return b_tau

    def _get_c_in(self, action: np.ndarray):
        """ 任务输入拆分 """
        allocated_size = [(self.function.input_size / self.function.difficulty) * a for a in action]
        c_r_in = allocated_size[:self.rsu_number]
        c_v_in = allocated_size[self.rsu_number:(self.rsu_number + self.vehicle_number)]
        c_c_in = allocated_size[-1]
        return c_r_in, c_v_in, c_c_in

    def _get_c_out(self, action: np.ndarray):
        """ 任务输出拆分 """
        allocated_size = [self.function.output_size * a for a in action]
        c_r_out = allocated_size[:self.rsu_number]
        c_v_out = allocated_size[self.rsu_number:(self.rsu_number + self.vehicle_number)]
        c_c_out = allocated_size[-1]
        return c_r_out, c_v_out, c_c_out

    def _take_action(self, c_r_in: List[float], c_v_in: List[float]) -> None:
        """ 如果满足卸载条件，则执行卸载动作 """
        for rsu, rsu_input in zip(self.rsu_list.rsu_list, c_r_in):
            rsu.task_in(rsu_input)

        for vehicle, vehicle_input in zip(self.vehicle_list.vehicle_list, c_v_in):
            vehicle.task_in(vehicle_input)

    def _tasklist_update(self) -> List[float]:
        """ 保证无关任务进入 """
        a_tau = [rsu.task_in(self.config.rsu_task_flow) for rsu in self.rsu_list.rsu_list]
        a_tau.extend(vehicle.task_in(self.config.vehicle_task_flow) for vehicle in self.vehicle_list.vehicle_list)
        return a_tau

    def _rsu_spent_time(self, c_r_in: List[float], c_r_out: List[float]) -> float:
        """ rsu部分最长执行时间 """
        rsu_time = []
        for i, rsu in zip(range(self.rsu_number), self.rsu_list.rsu_list):
            if c_r_in[i] == 0:
                rsu_time.append(0)
            else:
                rsu_queue = rsu.task_queue
                rsu_speed = rsu.compute_speed
                difficulty = self.function.difficulty
                w_r = rsu_queue / rsu_speed
                p_r = c_r_in[i] / (rsu_speed * difficulty)
                d_r = self.config.theta_rv * c_r_out[i]
                rsu_time.append((w_r + p_r + d_r))
        return max(rsu_time)

    def _vehicle_spent_time(self, c_v_in: List[float], c_v_out: List[float]):
        """ vehicle部分最长执行时间 """
        vehicle_time = []
        for i, vehicle in zip(range(self.vehicle_number), self.vehicle_list.vehicle_list):
            if c_v_in[i] == 0:
                vehicle_time.append(0)
            else:
                vehicle_queue = vehicle.task_queue
                vehicle_speed = vehicle.compute_speed
                difficulty = self.function.difficulty
                u_v = self.config.theta_rv * c_v_in[i]
                w_v = vehicle_queue / vehicle_speed
                p_v = c_v_in[i] / (vehicle_speed * difficulty)
                d_v = 2 * self.config.theta_rv * c_v_out[i]
                vehicle_time.append((u_v + w_v + p_v + d_v))
        return max(vehicle_time)

    def _cloud_spent_time(self, c_c_in: float, c_c_out: float):
        u_c = self.config.theta_rv * c_c_in
        d_c = self.config.theta_vc * c_c_out
        cloud_time = u_c + d_c
        return cloud_time

    def _spent_time(
            self,
            c_r_in: List[float],
            c_r_out: List[float],
            c_v_in: List[float],
            c_v_out: List[float],
            c_c_in: float,
            c_c_out: float
    ):
        """ 一个任务的总执行时间 """
        # rsu部分
        rsu_time = self._rsu_spent_time(c_r_in, c_r_out)
        # vehicle部分
        vehicle_time = self._vehicle_spent_time(c_v_in, c_v_out)
        # cloud部分
        cloud_time = self._cloud_spent_time(c_c_in, c_c_out)

        return max(rsu_time, vehicle_time, cloud_time)

    def _compute_y(self, a_tau, b_tau, c_r_in: List[float], c_v_in: List[float], tag):
        """ 计算y """
        if tag == "a":
            y_tau_rsu = [max((a_tau[i] + c_r_in[i] - b_tau[i]), 0) for i in range(self.rsu_number)]
            y_tau_vehicle = [max((a_tau[i + self.rsu_number] + c_v_in[i] - b_tau[i + self.rsu_number]), 0) for i in
                             range(self.vehicle_number)]
        else:
            y_tau_rsu = [max((c_r_in[i] - b_tau[i]), 0) for i in range(self.rsu_number)]
            y_tau_vehicle = [max((c_v_in[i] - b_tau[i + self.rsu_number]), 0) for i in range(self.vehicle_number)]
        return y_tau_rsu, y_tau_vehicle

    def _compute_B(self) -> float:
        """ B的计算 """
        B_r = sum((self.config.max_rsu_task_flow + self.config.max_in_datasize) ** 2
                  for _ in range(self.rsu_number)) / 2
        B_v = sum((self.config.max_vehicle_task_flow + self.config.max_in_datasize) ** 2
                  for _ in range(self.vehicle_number)) / 2
        B_tau = B_r + B_v
        return B_tau

    def _get_Q_tau(self):
        """ 获取时刻t的队列 """
        Q_tau_r = [rsu.task_queue for rsu in self.rsu_list.rsu_list]
        Q_tau_v = [vehicle.task_queue for vehicle in self.vehicle_list.vehicle_list]
        return Q_tau_r, Q_tau_v

    def _compute_growth(
            self,
            y_tau_rsu: List[float],
            y_tau_vehicle: List[float],
            B_tau: float,
            Q_tau_r: List[float],
            Q_tau_v: List[float]
    ):
        # 计算队伍增长量部分
        growth_r = sum(y_tau_rsu[i] * Q_tau_r[i] for i in range(self.rsu_number))
        growth_v = sum(y_tau_vehicle[j] * Q_tau_v[j] for j in range(self.vehicle_number))
        growth = growth_r + growth_v + B_tau
        return growth

    def _compute_backlog(
            self, a_tau, b_tau, c_r_in: List[float], c_v_in: List[float], B_tau: float, Q_tau_r: List[float],
            Q_tau_v: List[float]
    ):
        y_tau_rsu = [max((a_tau[i] + c_r_in[i] - b_tau[i]), 0) for i in range(self.rsu_number)]
        y_tau_vehicle = [max((a_tau[i + self.rsu_number] + c_v_in[i] - b_tau[i + self.rsu_number]), 0) for i in
                         range(self.vehicle_number)]
        backlog_r = sum(y_tau_rsu[i] * Q_tau_r[i] for i in range(self.rsu_number)) * (1 - 1 / (self.w + 10))
        backlog_v = sum(y_tau_vehicle[j] * Q_tau_v[j] for j in range(self.vehicle_number)) * (1 - 1 / (self.w + 10))
        backlog = backlog_r + backlog_v + B_tau
        return backlog

    def _Lyapunov_drift(self, Q_tau_r: List[float], Q_tau_r_: List[float], Q_tau_v: List[float], Q_tau_v_: List[float]):
        """ 增长 """
        drift_r = sum(((Q_tau_r_[i] ** 2) - (Q_tau_r[i] ** 2)) for i in range(self.rsu_number))
        drift_v = sum(((Q_tau_v_[i] ** 2) - (Q_tau_v[i] ** 2)) for i in range(self.vehicle_number))
        lyapunov_drift = (drift_r + drift_v) / 2
        return lyapunov_drift

    def _reward(
            self,
            a_tau: List[float],
            b_tau: List[float],
            delay: float,
            c_r_in: List[float],
            c_v_in: List[float],
            Q_tau_r: List[float],
            Q_tau_v: List[float]
    ):
        """ 计算reward """
        y_tau_rsu, y_tau_vehicle = self._compute_y(a_tau, b_tau, c_r_in, c_v_in, tag=self.stability)
        B_tau = self._compute_B()
        growth = self._compute_growth(y_tau_rsu, y_tau_vehicle, B_tau, Q_tau_r, Q_tau_v)
        backlog = self._compute_backlog(a_tau, b_tau, c_r_in, c_v_in, B_tau, Q_tau_r, Q_tau_v)
        lyapunov_object = growth + self.w * delay
        if delay > self.function.delay_threshold:
            reward = self.config.reward_threshold.get(self.weight_tag)
        else:
            reward = - lyapunov_object

        queue_v = sum(Q_tau_v) / self.vehicle_number
        queue_r = sum(Q_tau_r) / self.rsu_number
        queue = sum(Q_tau_r + Q_tau_v) / (self.rsu_number + self.vehicle_number)
        y_v = sum(y_tau_vehicle) / self.vehicle_number
        y_r = sum(y_tau_rsu) / self.rsu_number
        y = sum(y_tau_rsu + y_tau_vehicle) / (self.rsu_number + self.vehicle_number)

        return reward, backlog, queue_v, y_v, queue_r, y_r, queue, y

    def step(self, action):
        """ 状态转移 """
        c_r_in, c_v_in, c_c_in = self._get_c_in(action)
        c_r_out, c_v_out, c_c_out = self._get_c_out(action)
        Q_tau_r, Q_tau_v = self._get_Q_tau()  # 获取 t 时刻的队列长度
        b_tau = self._task_execute()  # 计算上个时隙完成的任务数据量
        self._take_action(c_r_in, c_v_in)  # 执行卸载动作
        delay = self._spent_time(c_r_in=c_r_in, c_r_out=c_r_out, c_v_in=c_v_in, c_v_out=c_v_out, c_c_in=c_c_in,
                                 c_c_out=c_c_out)  # 计算时间
        a_tau = self._tasklist_update()  # 保证其他任务进入
        reward, backlog, queue_v, y_v, queue_r, y_r, queue, y = self._reward(a_tau, b_tau, delay, c_r_in, c_v_in,
                                                                             Q_tau_r, Q_tau_v)
        self.vehicle_list.update()
        # 状态转移
        self.timeslot.add_time()
        done = self.timeslot.is_end()
        self.function = self._function_generator()
        self.state = self._rsu_perception()
        return np.array(self.state, dtype=np.float32), reward, backlog, delay, done, queue_v, y_v, queue_r, y_r, queue, y

    def render(self, mode='human'):
        # 不需要渲染，直接返回
        pass

    def close(self):
        pass
